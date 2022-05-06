from visualizers.roc_auc_visualizer import roc_curve_visualizer
from visualizers.personal_pred_visualizer import dkt_personal_pred_visualizer, dkvmn_personal_pred_visualizer

#모델별로 visualizers를 다르게 만들어야 함
#if 문으로 분기 주기
def get_visualizers(y_true_record, y_score_record,model, model_path, test_loader, device, config):

    if config.model_name == "dkt":
        roc_curve_visualizer(y_true_record, y_score_record, config.model_name)
        dkt_personal_pred_visualizer(model, model_path, test_loader, device, config.model_name)
    elif config.model_name == "dkvmn":
        roc_curve_visualizer(y_true_record, y_score_record, config.model_name)
        dkvmn_personal_pred_visualizer(model, model_path, test_loader, device, config.model_name)