from base import Transformation


class CustomTransformation(Transformation):
    def __init__(self, config, identifier_string):
        super().__init__(config, identifier_string)

    def add_customized_feature_engineering(self):
        self.df["created_to_launch"] = self.df["launched_at"] - self.df["created_at"]
        self.df["launch_to_deadline"] = self.df["deadline"] - self.df["launched_at"]
        self.df["created_to_deadline"] = self.df["deadline"] - self.df["created_at"]
        self.df["delta_state_changed_at"] = self.df["deadline"] - self.df["state_changed_at"]
        self.df.drop(["created_at", "launched_at", "state_changed_at", "deadline"], axis="columns", inplace=True)
