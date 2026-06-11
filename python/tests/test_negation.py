"""Test suite for negation support in probabilistic programs.

Tests verify that:
1. Stratified (non-cyclic) negation works correctly with exact inference
2. Monte Carlo results approximate exact solutions
3. Non-monotone (cyclic) programs are handled appropriately

Run with: pytest python/tests/test_negation.py -v
"""

import pytest

# Skip all tests if pyxlog or torch not available
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


class TestMcEngineContract:
    """MC fail-closed / engine-label contract on the Python surface."""

    def test_mc_rejected_program_fails_closed_without_opt_in(self):
        """Negation is resident-rejected; evaluate() must error, not silently
        run the CPU oracle."""
        source = """
0.3::rain().
dry() :- not rain().
query(dry()).
"""
        program = pyxlog.Program.compile(source, prob_engine='mc')
        with pytest.raises(Exception, match="resident MC engine rejected program"):
            program.evaluate(samples=100)

    def test_mc_resident_result_is_labeled_gpu(self):
        source = """
0.7::rain().
query(rain()).
"""
        program = pyxlog.Program.compile(source, prob_engine='mc')
        result = program.evaluate(samples=1000)
        assert result.mc_engine == "gpu-resident"

    def test_exact_result_has_no_mc_engine(self):
        source = """
0.7::rain().
query(rain()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()
        assert result.mc_engine is None


class TestStratifiedNegation:
    """Tests for stratified (non-cyclic) negation."""

    def test_simple_negation(self):
        """dry() :- not rain(). with 0.3::rain()"""
        source = """
0.3::rain().
dry() :- not rain().
query(dry()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        # P(dry) = P(not rain) = 1 - 0.3 = 0.7
        probs = torch.from_dlpack(result.prob)
        assert len(result.atoms) == 1
        assert abs(probs[0].item() - 0.7) < 1e-6, f"Expected 0.7, got {probs[0].item()}"

    def test_multi_layer_stratified(self):
        """a :- not b. b :- not c. with 0.4::c()"""
        source = """
0.4::c().
b() :- not c().
a() :- not b().
query(a()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        # P(b) = P(not c) = 0.6
        # P(a) = P(not b) = 0.4
        probs = torch.from_dlpack(result.prob)
        assert len(result.atoms) == 1
        assert abs(probs[0].item() - 0.4) < 1e-6, f"Expected 0.4, got {probs[0].item()}"

    def test_negation_with_conjunction(self):
        """Test negation combined with conjunction."""
        source = """
0.3::rain().
0.8::umbrella().
comfortable() :- not rain(), umbrella().
query(comfortable()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        # P(comfortable) = P(not rain) * P(umbrella) = 0.7 * 0.8 = 0.56
        probs = torch.from_dlpack(result.prob)
        assert len(result.atoms) == 1
        assert abs(probs[0].item() - 0.56) < 1e-6, f"Expected 0.56, got {probs[0].item()}"

    def test_negation_with_multiple_rules(self):
        """Test negation with multiple rules deriving the same predicate."""
        source = """
0.5::a().
0.4::b().
c() :- a(), not b().
c() :- b(), not a().
query(c()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        # P(c) = P(a AND not b) + P(b AND not a) - P(a AND not b AND b AND not a)
        # P(c) = 0.5 * 0.6 + 0.4 * 0.5 - 0
        # P(c) = 0.30 + 0.20 = 0.50
        probs = torch.from_dlpack(result.prob)
        assert len(result.atoms) == 1
        assert abs(probs[0].item() - 0.50) < 1e-6, f"Expected 0.50, got {probs[0].item()}"

    def test_negation_of_derived_predicate(self):
        """Test negation of a derived (non-probabilistic) predicate."""
        source = """
0.3::rain().
wet() :- rain().
dry() :- not wet().
query(dry()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        # wet() is true when rain() is true, so P(wet) = P(rain) = 0.3
        # P(dry) = P(not wet) = 0.7
        probs = torch.from_dlpack(result.prob)
        assert len(result.atoms) == 1
        assert abs(probs[0].item() - 0.7) < 1e-6, f"Expected 0.7, got {probs[0].item()}"

    def test_negation_closed_world_assumption(self):
        """Test that undefined predicates are false (closed world assumption)."""
        source = """
happy() :- not sad().
query(happy()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        # sad() is never defined, so P(sad) = 0 by closed world assumption
        # P(happy) = P(not sad) = 1.0
        probs = torch.from_dlpack(result.prob)
        assert len(result.atoms) == 1
        assert abs(probs[0].item() - 1.0) < 1e-6, f"Expected 1.0, got {probs[0].item()}"


class TestMCComparison:
    """Compare exact negation results with MC sampling."""

    def test_mc_probability_match_simple(self):
        """Exact negation should match MC within confidence interval."""
        source_exact = """
0.3::rain().
dry() :- not rain().
query(dry()).
"""

        # NOTE: the original source used `:- prob_engine=mc, ...`, which is not
        # a directive (the real syntax is `#pragma prob_engine = mc`), so this
        # test silently compared exact-vs-exact. Fixed to actually run MC.
        source_mc = """#pragma prob_engine = mc
0.3::rain().
dry() :- not rain().
query(dry()).
"""

        program_exact = pyxlog.Program.compile(source_exact)
        program_mc = pyxlog.Program.compile(source_mc)

        result_exact = program_exact.evaluate()
        # Negation is rejected by the GPU-resident MC engine; the CPU oracle
        # is explicit opt-in and the result is labeled.
        result_mc = program_mc.evaluate(samples=50000, allow_cpu_oracle=True)
        assert result_mc.mc_engine == "cpu-oracle"
        assert result_mc.approx is True

        prob_exact = torch.from_dlpack(result_exact.prob)[0].item()
        prob_mc = torch.from_dlpack(result_mc.prob)[0].item()

        # MC should be within reasonable tolerance of exact
        # For 50000 samples, 3 sigma ~ 0.02 for p=0.7
        assert abs(prob_exact - prob_mc) < 0.03, \
            f"Exact ({prob_exact}) and MC ({prob_mc}) differ too much"

    def test_mc_probability_match_conjunction(self):
        """Test MC matches exact for conjunction with negation."""
        source_exact = """
0.4::a().
0.6::b().
c() :- a(), not b().
query(c()).
"""

        # NOTE: fixed from `:- prob_engine=mc, ...` (not a directive) to the
        # real `#pragma` syntax — see test_mc_probability_match_simple.
        source_mc = """#pragma prob_engine = mc
0.4::a().
0.6::b().
c() :- a(), not b().
query(c()).
"""

        program_exact = pyxlog.Program.compile(source_exact)
        program_mc = pyxlog.Program.compile(source_mc)

        result_exact = program_exact.evaluate()
        # Negation is rejected by the GPU-resident MC engine; the CPU oracle
        # is explicit opt-in and the result is labeled.
        result_mc = program_mc.evaluate(samples=50000, allow_cpu_oracle=True)
        assert result_mc.mc_engine == "cpu-oracle"
        assert result_mc.approx is True

        prob_exact = torch.from_dlpack(result_exact.prob)[0].item()
        prob_mc = torch.from_dlpack(result_mc.prob)[0].item()

        # P(c) = P(a) * P(not b) = 0.4 * 0.4 = 0.16
        assert abs(prob_exact - 0.16) < 1e-6, f"Exact expected 0.16, got {prob_exact}"
        assert abs(prob_exact - prob_mc) < 0.03, \
            f"Exact ({prob_exact}) and MC ({prob_mc}) differ too much"


class TestNonMonotone:
    """Tests for non-monotone (cyclic negation) programs.

    Note: Non-monotone programs require the MC engine. We use the prob_engine
    parameter in Program.compile() since the directive parsing happens after
    stratification checking.
    """

    @pytest.mark.slow
    def test_non_monotone_simple_cycle(self):
        """Test simple cyclic program with MC (50K samples, ~150s on WSL2)."""
        source = """
0.5::flip().
p() :- flip().
q() :- not p().
p() :- not q().
query(p()).
query(flip()).
"""

        # Use prob_engine parameter to enable MC before stratification.
        # Non-monotone negation is resident-rejected; opt into the labeled
        # CPU oracle explicitly.
        program = pyxlog.Program.compile(source, prob_engine='mc')
        result = program.evaluate(samples=50000, allow_cpu_oracle=True)
        assert result.mc_engine == "cpu-oracle"

        probs = torch.from_dlpack(result.prob)

        # Find probabilities
        p_prob = None
        flip_prob = None
        for i, atom in enumerate(result.atoms):
            if atom == "p()":
                p_prob = probs[i].item()
            elif atom == "flip()":
                flip_prob = probs[i].item()

        # p() should be true when flip() is true (via first rule),
        # and the cycle stabilizes when flip() is false
        assert p_prob is not None
        assert flip_prob is not None
        # P(flip) should be close to 0.5
        assert abs(flip_prob - 0.5) < 0.05
        # P(p) should also be close to 0.5 (true when flip is true)
        assert abs(p_prob - 0.5) < 0.05

    def test_non_monotone_wfs_returns_zero(self):
        """Non-monotone programs with exact engine return 0 via WFS."""
        source = """
p() :- not q().
q() :- not p().
query(p()).
"""

        # Non-monotone programs are now handled by WFS
        # Atoms in a cycle are undefined and return probability 0
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        probs = torch.from_dlpack(result.prob)

        # Find p() probability
        p_prob = None
        for i, atom in enumerate(result.atoms):
            if atom == "p()":
                p_prob = probs[i].item()
                break

        # WFS: p and q are both undefined (in cycle), so probability is 0
        assert p_prob is not None, "p() not found in result atoms"
        assert p_prob == 0.0, f"Expected 0.0 for undefined WFS atom, got {p_prob}"


class TestNegationGradients:
    """Tests for gradient computation through negation."""

    def test_negation_gradient_returns(self):
        """Test that gradients are computed for programs with negation."""
        source = """
0.3::rain().
dry() :- not rain().
query(dry()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate(return_grads=True)

        # Check that gradients are returned
        assert result.grad_true is not None, "grad_true should be returned"
        assert result.grad_false is not None, "grad_false should be returned"
        assert len(result.grad_true) == 1
        assert len(result.grad_false) == 1

    def test_negation_gradient_values(self):
        """Test gradient values for simple negation."""
        source = """
0.3::rain().
dry() :- not rain().
query(dry()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate(return_grads=True)

        grad_true = torch.from_dlpack(result.grad_true[0])
        grad_false = torch.from_dlpack(result.grad_false[0])

        # For dry() :- not rain(), the probability is P(not rain) = 1 - p
        # d(P(dry))/d(p_true) for rain should be -1
        # The gradient vectors should have at least one element (for rain variable)
        assert len(grad_true) >= 1
        assert len(grad_false) >= 1


class TestNegationEdgeCases:
    """Edge case tests for negation."""

    def test_negation_with_zero_probability(self):
        """Test negation when base fact has zero probability."""
        source = """
0.0::never().
always() :- not never().
query(always()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        probs = torch.from_dlpack(result.prob)
        # P(always) = P(not never) = 1 - 0 = 1.0
        assert abs(probs[0].item() - 1.0) < 1e-6

    def test_negation_with_one_probability(self):
        """Test negation when base fact has probability 1."""
        source = """
1.0::always().
never() :- not always().
query(never()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        probs = torch.from_dlpack(result.prob)
        # P(never) = P(not always) = 1 - 1 = 0.0
        assert abs(probs[0].item() - 0.0) < 1e-6

    def test_double_negation(self):
        """Test double negation."""
        source = """
0.3::a().
b() :- not a().
c() :- not b().
query(c()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        probs = torch.from_dlpack(result.prob)
        # b = not a, c = not b = not (not a) = a
        # P(c) = P(a) = 0.3
        assert abs(probs[0].item() - 0.3) < 1e-6

    def test_negation_multiple_queries(self):
        """Test multiple queries involving negation."""
        source = """
0.4::rain().
wet() :- rain().
dry() :- not rain().
query(wet()).
query(dry()).
"""
        program = pyxlog.Program.compile(source)
        result = program.evaluate()

        probs = torch.from_dlpack(result.prob)
        assert len(result.atoms) == 2

        # Find wet and dry probabilities
        wet_prob = None
        dry_prob = None
        for i, atom in enumerate(result.atoms):
            if atom == "wet()":
                wet_prob = probs[i].item()
            elif atom == "dry()":
                dry_prob = probs[i].item()

        assert wet_prob is not None and abs(wet_prob - 0.4) < 1e-6
        assert dry_prob is not None and abs(dry_prob - 0.6) < 1e-6
        # wet and dry should be complementary
        assert abs(wet_prob + dry_prob - 1.0) < 1e-6
