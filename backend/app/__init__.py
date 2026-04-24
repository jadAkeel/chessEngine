"""Application package.

Kept intentionally light so importing subpackages such as ``app.infra`` does not
trigger heavyweight optional dependencies during tooling or unit tests.
"""

__all__ = [
    'api',
    'cli',
    'core',
    'evaluation',
    'game',
    'infra',
    'mcts',
    'model',
    'selfplay',
    'training',
]
