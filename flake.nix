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
            docker
            gdb
            gh
            llvmPackages.openmp
            lz4
            man-pages
            man-pages-posix
            neocmakelsp
            ninja
            nixd
            perf
            pkg-config
            tokei
            zstd
            # for viewing w neuroglancer
            python3
            uv
          ];
        };
      }
    );
}
