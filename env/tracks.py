"""Track seeds. Training uses fresh random tracks; these fixed sets are for selection and scoring."""

VALIDATION_SEEDS = list(range(100, 116))  # 16 tracks: model selection during training
TEST_SEEDS = list(range(2024, 2034))  # 10 tracks: the Play tab
GOAL_SEEDS = list(range(5000, 5100))  # 100 tracks: the goal check, never used for selection
