run_visual_html.shrun_visual.sh
#python display_error.py mengniu_error_list.txt mengniu_test_error_display.html mengniu_test_output
# alias python=/home/disk1/vis/wangxiaodi/tools/envs_mine/paddle_env_py37/bin/python3.7
root_dir='/root/paddlejob/workspace/env_run/output/haojing/GuidedDistillation-main-bak-0613/visual_res/'
data_dir='visual_gt'
data_list=${root_dir}"/"${data_dir}"_list.txt"
html_file=${root_dir}"/"${data_dir}"_display.html"
echo ${data_list}
echo ${html_file}

ls "${root_dir}/${data_dir}" > ${data_list}


/root/paddlejob/workspace/env_run/output/haojing/miniconda3_t_mamba/bin/python visual.py ${data_list} ${html_file} ${root_dir} ${data_dir}





