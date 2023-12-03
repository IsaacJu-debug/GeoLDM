#!/bin/bash
source start_ai_molecular_generation.sh

conda_prompt() {
  if [ -n "$CONDA_DEFAULT_ENV" ]; then
    # Use 'basename' to extract the last part of the path
    echo "($(basename $CONDA_DEFAULT_ENV))"
  else
    echo ""
  fi
}

# Set PS1 to use the conda_prompt function
export PS1='$(conda_prompt) \$ '

my_system=$(hostname | cut -d'.' -f1)
echo "jupyter lab --no-browser --port=5115 --ip=$my_system"

echo "ssh -L 127.0.0.1:5115:$my_system:5115 ju1@login.sherlock.stanford.edu" 
