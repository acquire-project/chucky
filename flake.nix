{
  description = "Development environment for chucky";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self, # required even if the lsp complains
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        formatter = pkgs.nixfmt-tree;

        devShells.default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
          name = "chucky";

          buildInputs = with pkgs; [
            cmake
            cudaPackages.cudatoolkit
            cudaPackages.nvcomp
            cudaPackages.nvcomp.static
            docker
            gdb
            gh
            llvmPackages.openmp
            (lz4.overrideAttrs (old: {
              cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-DBUILD_STATIC_LIBS=ON" ];
            }))
            man-pages
            man-pages-posix
            neocmakelsp
            ninja
            nixd
            perf
            pkg-config
            tokei
            (zstd.override { enableStatic = true; })
            # s3 writer
            aws-c-common
            aws-c-cal
            aws-c-io
            aws-c-http
            aws-c-auth
            aws-c-s3
            aws-c-compression
            aws-c-sdkutils
            aws-checksums
            s2n-tls
            awscli2
            # for viewing w neuroglancer
            python3
            uv
          ];
        };
      }
    );
}
