{
  description = "Development environment for chucky";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    claude-code.url = "github:sadjow/claude-code-nix";
    claude-code.inputs.nixpkgs.follows = "nixpkgs";
    claude-code.inputs.flake-utils.follows = "flake-utils";
  };

  outputs =
    {
      self, # required even if the lsp complains
      nixpkgs,
      flake-utils,
      claude-code,
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

          nativeBuildInputs = with pkgs; [
            cmake
            claude-code.packages.${system}.default
            docker
            gdb
            gh
            man-pages
            man-pages-posix
            neocmakelsp
            ninja
            nixd
            perf
            pkg-config
            tokei
            awscli2
            python3
            uv
          ];

          buildInputs = with pkgs; [
            cudaPackages.cudatoolkit
            cudaPackages.nvcomp
            cudaPackages.nvcomp.static
            llvmPackages.openmp
            (lz4.overrideAttrs (old: {
              cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-DBUILD_STATIC_LIBS=ON" ];
            }))
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
          ];
        };
      }
    );
}
