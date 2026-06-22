//! ST-TRC sound-(A): the epistemic (i)/(ii) discriminator accessors must be
//! PUBLIC so the road-not-taken derivation (xlog-runtime) can read per-atom
//! epistemic status cross-crate and apply the atom-level rejected filter.
//! This is a cross-crate test, so it fails to COMPILE if the accessors are
//! private — the TDD enforcement of the exposure.
use xlog_logic::ast::{Atom, Term};
use xlog_logic::epistemic::EpistemicInterpretation;

fn atom(predicate: &str, arity: usize) -> Atom {
    Atom {
        predicate: predicate.to_string(),
        terms: (0..arity as u32).map(Term::Symbol).collect(),
    }
}

#[test]
fn discriminator_accessors_classify_per_atom_cross_crate() {
    let interp = EpistemicInterpretation::new()
        .with_known("k", 1)
        .with_possible("p", 2)
        .with_rejected("r", 0);

    // class (i): possible = live hypothesis, not known, not rejected.
    assert!(interp.contains_possible(&atom("p", 2)));
    assert!(!interp.contains_rejected(&atom("p", 2)));
    assert!(!interp.contains_known(&atom("p", 2)));

    // class (ii): rejected = logic ruled it out. This is the atom-level filter
    // the (A) derivation applies to exclude (ii)-circular atoms.
    assert!(interp.contains_rejected(&atom("r", 0)));
    assert!(!interp.contains_possible(&atom("r", 0)));

    // known = committed (∀-worlds).
    assert!(interp.contains_known(&atom("k", 1)));

    // an unclassified atom is in none of the three sets.
    let other = atom("z", 1);
    assert!(!interp.contains_known(&other));
    assert!(!interp.contains_possible(&other));
    assert!(!interp.contains_rejected(&other));
}
