import timesfm

class InferlessPythonModel:
    def initialize(self):
        self.tfm = timesfm.TimesFm(
            context_len=128,
            horizon_len=96,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",
        )
        self.tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    def infer(self,inputs):
        forecast_input = inputs["forecast_input"]
        frequency_input = inputs["frequency_input"]
        point_forecast, experimental_quantile_forecast = self.tfm.forecast(
            [forecast_input],
            freq=[frequency_input],
        )
        return {
            "point_forecast":point_forecast,
            "experimental_quantile_forecast":experimental_quantile_forecast
        }

    def finalize(self):
        pass

