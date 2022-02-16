# seetaface6_onnx_model
convert origin csta model to onnx

# verified models
1. face_landmarker_pts5.csta - face_landmarker_pts5_net1/2.onnx 
2. face_landmarker_pts81.csta - face_landmarker_pts81_net1/2.onnx 
3. face_detector.csta - face_detector.onnx 
4. face_recognizer.csta - face_recognizer.onnx 

# not verified models
1. age_predictor.csta - age_predictor.onnx 
2. eye_state.csta - eye_state.onnx 
3. face_landmarker_mask_pts5.csta - face_landmarker_mask_pts5.onnx 
4. face_recognizer_light.csta - face_recognizer_light.onnx 
5. face_recognizer_mask.csta - face_recognizer_mask.onnx 
6. fas_first.csta - fas_first.onnx 
7. fas_second.csta - fas_second.onnx 
8. gender_predictor.csta - gender_predictor.onnx 
9. mask_detector.csta - mask_detector.onnx 
10. pose_estimation.csta - pose_estimation.onnx 
11. quality_lbn.csta - quality_lbn.onnx 

# not supported model
1. face_landmarker_mask_pts68.csta - face_landmarker_mask_pts68.onnx 

# modify model input shape
use test_scrpts/modify_model_input_shape.py to modify model shape  
./test_scrpts/modify_model_input_shape.py test.onnx input:1x3x480x480 save.onnx  

