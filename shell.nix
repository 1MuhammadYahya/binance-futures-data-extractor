let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    buildInputs = [
      pkgs.gcc
      pkgs.xclip
      pkgs.python3
      pkgs.python3Packages.virtualenv
      pkgs.python312Packages.numpy
      pkgs.python312Packages.pandas
      pkgs.python312Packages.pyperclip
    ];
    shellHook = ''
      [ -d venv ] || virtualenv venv
      source venv/bin/activate
      pip install binance-futures-connector==4.1.0 requests
    '';
  }
