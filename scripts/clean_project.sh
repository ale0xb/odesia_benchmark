#!/bin/bash
# Clean project by removing all files in csvs and trained_models directories

# Check if csvs directory exists
if [ -d "csvs" ]; then
  # Remove all files in csvs directory
  rm -rf csvs/*
else
  echo "Directory csvs does not exist."
fi

# Check if trained_models directory exists
if [ -d "trained_models" ]; then
  # Remove all files in trained_models directory
  rm -rf trained_models/*
else
  echo "Directory trained_models does not exist."
fi

# Remove report.json
rm -rf report.json

