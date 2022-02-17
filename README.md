# seetaface6_onnx_model
convert origin csta model to onnx

# verified models
1. face_landmarker_pts5.csta - face_landmarker_pts5_net1/2.onnx 
2. face_landmarker_pts81.csta - face_landmarker_pts81_net1/2.onnx 
3. face_detector.csta - face_detector.onnx 
4. face_recognizer.csta - face_recognizer.onnx 
5. age_predictor.csta - age_predictor.onnx 

# not verified models
1. eye_state.csta - eye_state.onnx 
2. face_landmarker_mask_pts5.csta - face_landmarker_mask_pts5.onnx 
3. face_recognizer_light.csta - face_recognizer_light.onnx 
4. face_recognizer_mask.csta - face_recognizer_mask.onnx 
5. fas_first.csta - fas_first.onnx 
6. fas_second.csta - fas_second.onnx 
7. gender_predictor.csta - gender_predictor.onnx 
8. mask_detector.csta - mask_detector.onnx 
9. pose_estimation.csta - pose_estimation.onnx 
10. quality_lbn.csta - quality_lbn.onnx 

# not supported model
1. face_landmarker_mask_pts68.csta - face_landmarker_mask_pts68.onnx 

# modify model input shape
use test_scrpts/modify_model_input_shape.py to modify model shape  
./test_scrpts/modify_model_input_shape.py test.onnx input:1x3x480x480 save.onnx  

