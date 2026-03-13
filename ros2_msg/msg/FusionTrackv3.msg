#state : 27(ver.)
std_msgs/Header header
# Measure (12)
float64 measure_rel_pos_y
float64 measure_rel_pos_x
float64 measure_rel_vel_y
float64 measure_rel_vel_x
float64 measure_abs_vel_y
float64 measure_abs_vel_x
float64 measure_abs_vel
float64 measure_rel_acc_y
float64 measure_rel_acc_x
float64 measure_width
float64 measure_length
float64 measure_heading_angle

# Assocation (8)
float64 association_fused_fvt_id
float64 association_fused_ldt_id
float64 association_fused_frt_id
float64 association_asso_hist_fvt_id
float64 association_asso_hist_ldt_id
float64 association_asso_hist_frt_id
float64 association_asso_creation_flag
float64 association_asso_association_flag


# Motion Attribute (2)
float64 motion_attribute_abs_vel
float64 motion_attribute_motion

# Shape (1)
float64 shape_attribute_shape

# Tracking (19)
float64 tracking_id
float64 tracking_rel_pos_y
float64 tracking_rel_pos_x
float64 tracking_rel_vel_y
float64 tracking_rel_vel_x
float64 tracking_rel_acc_y
float64 tracking_rel_acc_x
float64 tracking_updated_age
float64 tracking_coasting_age
float64 tracking_life_time
float64 tracking_rel_pos_normal
float64 tracking_rel_vel_tangential
float64 tracking_rel_vel_normal
float64 tracking_abs_vel_x
float64 tracking_abs_vel_y
float64 tracking_moving_direction
float64 tracking_heading_angle
float64 tracking_range
float64 tracking_angle

# Thread (13)
float64 tracking_threat_p_long
float64 tracking_threat_ttc
float64 tracking_threat_d_br
float64 tracking_threat_d_w
float64 tracking_threat_x_p
float64 tracking_threat_i_long
float64 tracking_threat_dlc
float64 tracking_threat_tlc
float64 tracking_threat_i_lat
float64 tracking_threat_rss_x
float64 tracking_threat_rss_y
float64 tracking_threat_fcw_honda
float64 tracking_threat_cp

# Vehicle Recognition (1)
float64 vehicle_recognition

# Vehicle Manuever (2)
float64 vehicle_manuever_pr_ctrv
float64 vehicle_manuever

# Pedestrian Manuever (1)
float64 pedestrian_recognition

# Cyclist Recogntion
float64 cyclist_recognition

# E_Scooter Recognition
float64 e_scooter_recognition
