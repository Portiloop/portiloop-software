with import <nixpkgs> {};
mkShell {
  name = "tensorflow-cuda-shell";
  buildInputs = with python3.pkgs; [
    pip
    setuptools
    python39Packages.tensorflow
    python39Packages.numpy
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' TMPDIR='$HOME' \pip"
    export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.7/site-packages:$PYTHONPATH"
    export PATH="$(pwd)/_build/pip_packages/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}
