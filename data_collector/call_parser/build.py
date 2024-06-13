# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `call_parser` directory
    'my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-python',
    'tree-sitter-typescript/tsx',
    'tree-sitter-typescript/typescript',
    'tree-sitter-java',
    'tree-sitter-javascript',
  ]
)