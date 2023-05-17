def detect_landmarks(inputs, model_ft):
    outputs, _ = model_ft(inputs)
    pred_heatmap = outputs[-1][:, :-1, :, :]
    return pred_heatmap[:, 96, :, :], pred_heatmap[:, 97, :, :]
