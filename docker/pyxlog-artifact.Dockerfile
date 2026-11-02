FROM scratch
LABEL org.opencontainers.image.source="https://github.com/BrainyBlaze/xlog" \
    org.opencontainers.image.description="Maintainer-built Python wheel and original build records"
COPY --chmod=0444 . /xlog-release/
