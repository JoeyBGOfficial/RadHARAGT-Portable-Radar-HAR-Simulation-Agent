%% RadHARSimulator V3 Main Script
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2026-04-01.
% Affiliate: Beijing Institute of Technology.
% Platform: Ollama, MATLAB R2025b, Python 3.10.13 with Conda EMDM Environment.
%
% Introduction:
%   This script implements a comprehensive pipeline for generating realistic indoor human radar echo datasets.
%   It utilizes a large language model via a local REST API to interpret user descriptions and extract precise radar simulation parameters.
%   The script refines the input into an optimized prompt to drive a python-based motion diffusion model environment.
%   It intercepts the newly generated human motion sequence and converts it into a native workspace array.
%   The motion sequence undergoes signal post-processing including Savitzky-Golay filtering and variance-based idle frame truncation.
%   The script simulates frequency modulated continuous wave radar point-scatterer echoes based on the three-dimensional trajectories and physical radar equations.
%   It computes the direct path and ground multipath reflections utilizing predefined normalized radar cross section values for human skeletal joints.
%   The system calculates the Range-Time Matrix and Doppler-Time Matrix using fast Fourier transforms and short-time Fourier transforms.
%   It optionally employs a pre-trained deep convolutional neural network to reduce background noise from the spectral representations.
%   The script exports high-resolution logarithmic magnitude images tailored for training deep learning models.
%   Finally, it renders an animated three-dimensional visualization of the human skeleton motion within the simulated physical scene.
%
% References:
% [1] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-Or, and A. Shamir, "Human motion diffusion model," in Proc. Int. Conf. Learn. Represent., Kigali, Rwanda, May 2023.
% [2] V. C. Chen, F. Li, S. -S. Ho, and H. Wechsler, "Micro-Doppler effect in radar: phenomenon, model, and simulation study," IEEE Trans. Aerosp. Electron. Syst., vol. 42, no. 1, pp. 2-21, Jan. 2006.
% [3] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, "Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising," IEEE Trans. Image Process., vol. 26, no. 7, pp. 3142-3155, Jul. 2017.
%
% Dependencies:
%   MATLAB R2025b.
%   Deep Learning Toolbox.
%   Signal Processing Toolbox.
%   Image Processing Toolbox.
%   Python 3.10 environment with EMDM installed.
%   Ollama local deployment configured with qwen3-vl:4b model suggested.

%% Preparation for Matlab Script
close all;
clear all;
clc;

% Replace default print with agent thinking stream
agent_think("Author: JoeyBG. RadHARAGT system initialized.");

%% Definition
% Define the basic parameters for the simulation
Model_Name = "qwen3-vl:4b";                                                 % Name of the LLM model used for prompt refining and information extracting                                                 
Input_Text = "我现在需要仿真一个人绕圈步行的行为。采用2GHz中心频率、1GHz带宽的超宽带雷达，PRF设置为400Hz，发射天线位置设置为[0, 5, 1]，接收天线位置设置为[0.25, 5, 1]，天线增益设置为11dBi，系统信噪比设置为50dB。场景中需要启用墙体，墙体位置设置为[0, 4.5, 1]，墙体宽度5米、高度3米、厚度0.2米，墙体介电常数设置为4.5、损耗角正切设置为0.05。仿真过程需要考虑多径效应，但不使用神经网络进行特征增强。";                             
Has_Image = false;                                                          % Define whether the LLM use image for input                                                      
Image_Path = "Reference.jpg";                                               % Path of the input image                                              

% Define the parameters for the motion generation execution
EMDM_Path = "EMDM";                                                         % Path name of the EMDM human motion model generation project                                                  
Conda_Env = "emdm";                                                         % Environment name of the EMDM project
Guidance_Param = "5.0";                                                     % Guidance parameter of EMDM. Larger value means more compliance to the input prompt but less creative
Motion_FPS = 20;                                                            % FPS information of the EMDM output

% Define the visualization style parameters
Font_Name = 'Palatino Linotype';                                            % Font name of the visualization                                           
Font_Size_Basis = 15;                                                       % Basic font size of the visualization
Font_Size_Axis = 16;                                                        % Axis font size of the visualization
Font_Size_Title = 18;                                                       % Title font size of the visualization
Font_Weight_Basis = 'normal';                                               % Basic font weight of the visualization
Font_Weight_Axis = 'normal';                                                % Axis font weight of the visualization
Font_Weight_Title = 'bold';                                                 % Title font weight of the visualization

% Define the custom colormap colors for the visualization plots
JoeyBG_Colormap =[0.6196 0.0039 0.2588;                                     
                   0.8353 0.2431 0.3098;
                   0.9569 0.4275 0.2627;
                   0.9922 0.6824 0.3804;
                   0.9961 0.8784 0.5451;
                   1.0000 1.0000 0.7490;
                   0.9020 0.9608 0.5961;
                   0.6706 0.8667 0.6431;
                   0.4000 0.7608 0.6471;
                   0.1961 0.5333 0.7412;
                   0.3686 0.3098 0.6353];                                   % My favorite colormap
JoeyBG_Colormap_Flip = flip(JoeyBG_Colormap);                               % Flip the custom colormap for alternate visualizations                               

% Create a timestamped folder to save all simulation outputs
current_time_str = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));      % Use current running time for naming
Save_Dir = sprintf('Simulation_Result_%s', current_time_str);
if ~exist(Save_Dir, 'dir')
    mkdir(Save_Dir);
end

% Replace default print with agent thinking stream
agent_think(sprintf("Allocating local workspace directory at %s to securely store all generated simulation outputs.", Save_Dir));

% Define the system prompt template to instruct the model for dual task extraction
JSON_Template = sprintf(['{\n', ...
    '  "Refined_Prompt": "...",\n', ...
    '  "fc": 77000000000,\n', ...
    '  "tp": 0.001,\n', ...
    '  "B": 4000000000,\n', ...
    '  "PRF": 1000,\n', ...
    '  "fs": 10000000,\n', ...
    '  "tx_pos": [0, 2, 1],\n', ...
    '  "rx_pos":[0, 2, 1],\n', ...
    '  "antenna_gain": 15,\n', ...
    '  "antenna_isolation": 40,\n', ...
    '  "SNR": 50,\n', ...
    '  "enable_wall": false,\n', ...
    '  "enable_multipath": true,\n', ...
    '  "wall_center":[0, 1, 1],\n', ...
    '  "wall_dimensions":[3, 0.2, 2.5],\n', ...
    '  "wall_epsilon_r": 4.5,\n', ...
    '  "wall_loss_tangent": 0.05,\n', ...
    '  "stft_window_seconds": 0.1,\n', ...
    '  "stft_overlap_ratio": 0.75,\n', ...
    '  "enable_network": true\n', ...
    '}']);

% Construct the full text instruction for the language model
% The system prompt consists of three different parts:
%   1. Define the task of the LLM model we used in Chinese
%   2. Use LLM to refine the input prompt for human motion model generation
%   3. Extract key radar and scene parameters for simulation
System_Prompt = strcat("你现在是一个专业的Human Motion生成与雷达仿真物理参数提取专家。\n"...
    , "你的核心任务是理解用户对'室内人体动作与雷达探测场景'的自然语言描述，并将其精准地转化为结构化的JSON参数字典。\n\n"...
    , "【任务一：动作语义转换 (Refined_Prompt)】\n"...
    , "将用户的基础动作描述翻译并润色为适合HumanML3D/MDM生成3D序列的英文Prompt（10-30词）。对于跌倒、坐下等快速且有明确终态的动作，必须明确指出速度和最终姿态（例如：quickly falls down and remains lying on the ground）。\n\n"...
    , "【任务二：雷达与物理场景参数提取】\n"...
    , "从描述中提取雷达射频参数、天线位置及场景媒质属性。请严格遵循以下物理单位和坐标系定义（若用户未提供某些参数，请根据典型的室内FMCW/UWB人体雷达探测经验补充合理默认值）：\n"...
    , "- 射频参数（必须转换为基本单位）：中心频率 fc (Hz，例如2GHz必须输出为 2000000000)、带宽 B (Hz，例如1GHz为 1000000000)、脉冲宽度 tp (秒, 默认 1e-3)、脉冲重复频率 PRF (Hz)、采样率 fs (Hz, 默认 10e6)。\n"...
    , "- 系统与天线：系统信噪比 SNR (dB, 默认 50)、天线增益 antenna_gain (dBi, 默认 15)、收发隔离度 antenna_isolation (dB, 默认 40)。\n"...
    , "- 坐标系定义（绝对准则）：所有的三维向量必须严格遵守 [X(方位/宽度), Y(距离/深度/厚度), Z(高度)] 的笛卡尔物理坐标系。\n"...
    , "- 天线位置：发射天线 tx_pos 和接收天线 rx_pos 格式为 [X, Y, Z] (单位：米)。\n"...
    , "- 墙体参数（重要）：若语义中包含墙体则 enable_wall = true，墙体中心 wall_center 为 [X, Y, Z]。**墙体尺寸 wall_dimensions 必须且只能按照 [X轴宽度, Y轴厚度, Z轴高度] 的顺序输出！** 例如用户说'宽5米、高3米、厚0.2米'，你必须输出为 [5.0, 0.2, 3.0]！墙体介电常数 wall_epsilon_r 和损耗角正切 wall_loss_tangent 按原意提取。\n"...
    , "- 信号处理：stft_window_seconds (秒, 默认 0.1)，stft_overlap_ratio (0~1之间, 默认 0.75)。\n"...
    , "- 功能开关 (布尔值 true/false)：是否启用墙体 (enable_wall)、是否考虑多径效应 (enable_multipath)、是否使用神经网络进行特征增强/去噪 (enable_network)。如果用户明确说不使用某项功能，设为 false。\n\n"...
    , "【输出格式严格约束】\n"...
    , "你的输出必须且只能是一个合法的JSON对象，完全匹配下方模板的键值名称和数据类型。绝对不要输出任何额外的思考过程、问候语或 ```json 等Markdown标记！\n"...
    , "模板示例：\n", JSON_Template);

%% Connect to Ollama and Generate Prompt & Parameters
% Replace default print with agent thinking stream
agent_think("Initiating connection to the local large language model to comprehend the scene description and extract required simulation parameters.");

% Combine the system prompt and user input text
Combined_Text = strcat(System_Prompt, newline, newline, "【用户的初始描述】:", Input_Text);

try
    if Has_Image
        if isfile(Image_Path)
            % Replace default print with agent thinking stream
            agent_think(sprintf("Reading and encoding the specified reference visual material from %s into base64 format for multimodal analysis.", Image_Path));
            
            % Read the reference image and encode it to base64 format
            fid = fopen(Image_Path, 'rb');
            imgData = fread(fid, '*uint8');
            fclose(fid);
            imgBase64 = matlab.net.base64encode(imgData);
            
            % Set up the standard REST API request options for the vision model
            apiEndpoint = "http://localhost:11434/api/generate";
            options = weboptions('MediaType', 'application/json', 'Timeout', 120);
            
            % Construct the payload containing model configuration and prompt and encoded images
            payload = struct(...
                'model', Model_Name, ...
                'prompt', Combined_Text, ...
                'stream', false, ...
                'images', {cellstr(imgBase64)} ... 
            );
            
            % Replace default print with agent thinking stream
            agent_think("Transmitting the multimodal request payload to the local language model and awaiting cognitive response.");
            
            % Send the request to the local model and store the response
            response = webwrite(apiEndpoint, payload, options);
            LLM_Raw_Response = response.response;
        else
            error("Error. The specified reference image file does not exist.");
        end
    else
        % Replace default print with agent thinking stream
        agent_think("No visual reference material was detected. Proceeding exclusively with text based semantic reasoning.");
        
        % Use the built in chat object to generate response for text only input
        chat = ollamaChat(Model_Name);
        [responseMsg, chat] = generate(chat, Combined_Text);
        LLM_Raw_Response = responseMsg; 
    end

    %% Parse JSON to Initialize Simulation_Params
    % Replace default print with agent thinking stream
    agent_think("Intercepted communication from the language model. Decoding the structured data payload to extract physical variables.");
    
    % Clean up markdown code block strings from the raw output
    json_str = strtrim(string(LLM_Raw_Response));
    
    % Robust JSON extraction overriding random LLM conversational prefix/suffix text securely
    idx_first = strfind(json_str, '{');
    idx_last = strfind(json_str, '}');
    if ~isempty(idx_first) && ~isempty(idx_last)
        json_str = extractBetween(json_str, idx_first(1), idx_last(end), 'Boundaries', 'inclusive');
    else
        json_str = regexprep(json_str, '^```json\s*', '');
        json_str = regexprep(json_str, '^```\s*', '');
        json_str = regexprep(json_str, '\s*```$', '');
    end
    
    try
        % Decode the clean json string into the simulation parameters structure
        Simulation_Params = jsondecode(json_str);
    catch ME
        % Replace default print with agent thinking stream and standard format
        agent_think("Encountered a critical structural failure while attempting to decode the format of the response. Printing the raw output text below for physical diagnostics.");
        fprintf('%s\n', json_str);
        error("The language model did not return a valid structured format.");
    end
    
    % Ensure basic JSON fields exist avoiding unexpected LLM output omissions structurally crashing the system
    Default_Params = struct('fc', 77e9, 'tp', 1e-3, 'B', 4e9, 'PRF', 4000, ...
        'fs', 10e6, 'tx_pos', [0, 5, 1], 'rx_pos', [0, 5, 1], 'antenna_gain', 15, ...
        'antenna_isolation', 40, 'SNR', 50, 'enable_wall', false, 'enable_multipath', true, ...
        'wall_center', [0, 1, 1], 'wall_dimensions', [3, 0.2, 2.5], ...
        'wall_epsilon_r', 4.5, 'wall_loss_tangent', 0.05, ...
        'stft_window_seconds', 0.1, 'stft_overlap_ratio', 0.75, 'enable_network', false);
    
    fields_list = fieldnames(Default_Params);
    for idx_field = 1:length(fields_list)
        if ~isfield(Simulation_Params, fields_list{idx_field})
            Simulation_Params.(fields_list{idx_field}) = Default_Params.(fields_list{idx_field});
        end
    end
    
    % Validate the existence of the refined prompt field
    if isfield(Simulation_Params, 'Refined_Prompt')
        Refined_Prompt = Simulation_Params.Refined_Prompt;
    else
        error("The parsed data structure is missing the refined prompt field.");
    end

    % Replace default print with agent thinking stream
    agent_think("Successfully extracted radar hardware properties and refined the semantic text prompt for the downstream framework.");
    
    %% Call EMDM to Generate 3D Motion Data
    % Replace default print with agent thinking stream
    agent_think("Delegating the three dimensional human motion trajectory generation to the deep diffusion mathematical framework utilizing the newly optimized prompt.");
    
    % Clean the generated prompt text for command line usage
    Cleaned_Prompt = string(Refined_Prompt);
    Cleaned_Prompt = strrep(Cleaned_Prompt, newline, ' ');
    Cleaned_Prompt = strrep(Cleaned_Prompt, '"', '');
    Cleaned_Prompt = strrep(Cleaned_Prompt, '''', '');
    
    % Save the original directory path and switch to the motion generation directory
    original_dir = pwd;
    cd(EMDM_Path);
    
    % Scan the current directory to track existing results files
    pre_run_files = dir(fullfile('models', '**', 'results.npy'));
    pre_run_paths = string(fullfile({pre_run_files.folder}, {pre_run_files.name}));
    
    % Construct the execution command for the conda environment
    model_pth = fullfile('models', 'HumanML3D.pth');
    cmd = sprintf('conda run -n %s python sample_mdm.py --text_prompt "%s" --model_path %s --dataset humanml --guidance_param %s', ...
                  Conda_Env, Cleaned_Prompt, model_pth, Guidance_Param);
                  
    % Replace default print with agent thinking stream
    agent_think(sprintf("Executing system shell command to trigger the remote python inference kernel and synthesize the complex action progression. The command is %s", cmd));
    
    % Run the motion generation command through the system shell
    system(cmd);
    
    % Replace default print with agent thinking stream
    agent_think("The motion progression sequence has been successfully compiled and synthesized by the external python computation environment.");
    
    %% Find & Load Generated Results
    % Scan the directory again to locate the newly generated results
    post_run_files = dir(fullfile('models', '**', 'results.npy'));
    post_run_paths = string(fullfile({post_run_files.folder}, {post_run_files.name}));
    newly_generated_files = setdiff(post_run_paths, pre_run_paths);
    
    % Throw an error if no new motion data file is found
    if isempty(newly_generated_files)
        cd(original_dir);
        % Pre-error agent thought
        agent_think("Action generation failed. No valid resulting matrix was detected within the system pipeline.");
        error("Action generation failed. No new resulting array was generated. Skipping visualization.");
    end
    
    % Select the most recent numpy file from the detected new files
    latest_npy = char(newly_generated_files(end));
    
    % Define temporary script names for the python conversion tool
    temp_py = 'temp_npy2mat.py';
    temp_mat = 'temp_motion.mat';
    
    % Write the python script to safely convert the numpy array into a matlab readable format
    fid_py = fopen(temp_py, 'w');
    fprintf(fid_py, 'import sys\nimport numpy as np\nimport scipy.io as sio\n');
    fprintf(fid_py, 'npy_path = sys.argv[1]\nmat_path = sys.argv[2]\n');
    fprintf(fid_py, 'data = np.load(npy_path, allow_pickle=True)\n');
    fprintf(fid_py, 'if isinstance(data, np.ndarray) and data.shape == (): data = data.item()\n');
    fprintf(fid_py, 'motion = data.get("motion", data) if isinstance(data, dict) else data\n');
    fprintf(fid_py, 'motion = np.squeeze(motion)\n');
    fprintf(fid_py, 'while len(motion.shape) > 3: motion = motion[0]\n');
    fprintf(fid_py, 'if motion.shape[0] == 22 and motion.shape[1] == 3: motion = np.transpose(motion, (2, 0, 1))\n');
    fprintf(fid_py, 'sio.savemat(mat_path, {"motion": motion})\n');
    fclose(fid_py);
    
    % Execute the python conversion script inside the conda environment
    conv_cmd = sprintf('conda run -n %s python %s "%s" "%s"', Conda_Env, temp_py, latest_npy, temp_mat);
    system(conv_cmd);
    
    % Embed a rigorous file generation verification preventing silent execution failures crashing subsequent loadings
    if ~isfile(temp_mat)
        if isfile(temp_py), delete(temp_py); end
        cd(original_dir);
        % Pre-error agent thought
        agent_think("Data format conversion pipeline collapsed. Python process failed to bridge external structures.");
        error("Python conversion failed to create the target temporary data file. Please investigate python and deep learning environments.");
    end
    
    % Load the converted motion matrix data into the workspace
    loaded_data = load(temp_mat);
    motion_mat = double(loaded_data.motion); 
    
    % Delete the temporary files and restore the original directory context
    delete(temp_py);
    delete(temp_mat);
    cd(original_dir); 
    
    %% Post-Processing to Improve Motion Quality
    % Replace default print with agent thinking stream
    agent_think("Applying high order kinematic smoothing filters and calculating variance based temporal truncation limits to stabilize the raw structural geometry.");
    
    % Apply filter parameters to smooth the chaotic motion trajectories
    window_size = 9; 
    poly_order = 3;  
    if size(motion_mat, 1) > window_size
        for j = 1:size(motion_mat, 2)
            for d = 1:size(motion_mat, 3)
                motion_mat(:, j, d) = sgolayfilt(motion_mat(:, j, d), poly_order, window_size);
            end
        end
    end
    
    % Calculate relative velocity to find the active frames
    orig_frames = size(motion_mat, 1);
    root_pos = motion_mat(:, 1, :);
    rel_motion = motion_mat - root_pos;
    rel_vel = diff(rel_motion, 1, 1); 
    
    % Calculate the overall frame activity energy to detect the idle state
    frame_activity = sum(sum(rel_vel.^2, 2), 3); 
    idle_threshold = 1e-4; 
    is_active = frame_activity > idle_threshold;
    
    % Find the last frame where the human skeleton remains active
    last_active_frame = orig_frames;
    for f = orig_frames-1 : -1 : 1
        if is_active(f)
            last_active_frame = f + 1;
            break;
        end
    end
    
    % Truncate the motion matrix to prevent drifting after the action finishes
    pad_frames = 5;
    cut_frame = min(last_active_frame + pad_frames, orig_frames);
    if cut_frame < orig_frames - 10 
        motion_mat = motion_mat(1:cut_frame, :, :);
    end
    
    %% Build and Save the Comprehensive Simulation_Params Struct
    % Replace default print with agent thinking stream
    agent_think("Consolidating the extracted hardware properties and spatial vector coordinates into the persistent physical parameter collection system.");
    
    % Append the refined and truncated three dimensional motion matrix
    Simulation_Params.motion_mat = motion_mat;                     
    Simulation_Params.num_frames = size(motion_mat, 1);            
    Simulation_Params.num_joints = size(motion_mat, 2);            
    
    % Append temporal information crucial for the doppler shift calculation
    Simulation_Params.FPS = Motion_FPS;                            
    Simulation_Params.time_axis = (0 : Simulation_Params.num_frames - 1) / Motion_FPS; 
    
    % Define the standard kinematic tree topology for bone connecting
    kinematic_tree =[
        1, 2; 1, 3; 1, 4; 2, 5; 3, 6; 4, 7; 5, 8; 6, 9; 
        7, 10; 8, 11; 9, 12; 10, 13; 10, 14; 10, 15; 13, 16; 
        14, 17; 15, 18; 17, 19; 18, 20; 19, 21; 20, 22
    ];
    Simulation_Params.kinematic_tree = kinematic_tree;
    
    % Define the normalized radar cross section values for each body joint
    Normalized_RCS =[1.0, 0.7, 0.7, 0.9, 0.5, 0.5, 0.8, 0.3, 0.3, 0.8, ...
                      0.1, 0.1, 0.6, 0.5, 0.5, 0.85, 0.4, 0.4, 0.3, 0.3, ...
                      0.1, 0.1];
    Simulation_Params.Normalized_RCS = Normalized_RCS;
    
    % Map the normalized values to spatial physical radii for the drawing templates
    min_joint_radius = 0.02;
    max_joint_radius = 0.075;
    joint_radii = min_joint_radius + (max_joint_radius - min_joint_radius) * Normalized_RCS;
    
    % Store the calculated radii settings into the main structure
    Simulation_Params.joint_radii = joint_radii;                   
    Simulation_Params.bone_radius = 0.035;                         
    
    % Save the fully packed simulation parameters struct into the current directory
    save(fullfile(Save_Dir, 'Simulation_Params.mat'), 'Simulation_Params');

    %% FMCW Radar Point-Scatterer Simulation
    % Replace default print with agent thinking stream
    agent_think("Engaging the frequency modulated continuous wave radar physics engine to computationally simulate all internal point scatterer interference echoes and wall reflections.");
    
    % Extract the radar parameters from the simulation settings
    c = 3e8;                                            
    fc = Simulation_Params.fc;
    lambda = c / fc;                                    % Parse the radar wavelength for proper decay calculations
    tp = Simulation_Params.tp;
    B = Simulation_Params.B;
    K = B / tp;                                         
    PRF = Simulation_Params.PRF;
    fs = Simulation_Params.fs;
    
    % Incorporate antenna gain calculating absolute physics scaled amplitudes
    if isfield(Simulation_Params, 'antenna_gain')
        G_lin = 10^(Simulation_Params.antenna_gain / 10);
    else
        G_lin = 10^(10 / 10);                           % Fallback default generic indoor 10dBi patch array
    end
    
    % Establish fundamental base amplitude factor complying strictly with the physical bistatic radar equation
    base_amp_factor = (lambda * G_lin) / ((4 * pi)^1.5);
    
    % Calculate fundamental electromagnetic parameters for physical wall penetration logic
    epsilon_0 = 8.854e-12;
    mu_0 = 4 * pi * 1e-7;
    omega = 2 * pi * fc;

    % Initialize default wall variables to prevent undefined errors when the wall feature is disabled
    wall_alpha = 0.0;
    v_wall = c;
    wall_gamma = 0.0;
    wall_y_center = 0.0;
    wall_y_thickness = 0.0;
    wall_y_min = 0.0;
    wall_y_max = 0.0;
    
    if Simulation_Params.enable_wall
        eps_r = Simulation_Params.wall_epsilon_r;
        loss_tan = Simulation_Params.wall_loss_tangent;
        
        eps_real = eps_r * epsilon_0;
        eps_imag = eps_real * loss_tan;
        
        % Calculate wave attenuation constant alpha (Np/m) based on physical properties
        wall_alpha = omega * sqrt(mu_0 * eps_real / 2 * (sqrt(1 + (eps_imag/eps_real)^2) - 1));
        
        % Calculate propagation velocity within the specific wall medium
        v_wall = c / sqrt(eps_r);
        
        % Calculate the intrinsic reflection coefficient (Gamma) for transmission limits
        % Keep correct reflection polarity restoring the pi phase shift physics upon dielectric reflection
        wall_gamma = (1 - sqrt(eps_r)) / (1 + sqrt(eps_r));
        
        % Parse geometrical coordinates boundaries for spatial intersection computations
        wall_y_center = Simulation_Params.wall_center(2);
        wall_y_thickness = Simulation_Params.wall_dimensions(2);
        wall_y_min = wall_y_center - wall_y_thickness / 2;
        wall_y_max = wall_y_center + wall_y_thickness / 2;
    end
    
    % Extract the spatial coordinates of the transmitting and receiving antennas
    tx_pos = Simulation_Params.tx_pos(:)';              
    rx_pos = Simulation_Params.rx_pos(:)';
    
    % Configure the fast time samples and interval duration
    N_s = round(tp * fs);                               
    t_fast = (0:N_s-1)' / fs;
    
    % Calculate the slow time axis based on the pulse repetition frequency
    t_motion = Simulation_Params.time_axis;             
    t_slow = 0 : 1/PRF : t_motion(end);                 
    num_pulses = length(t_slow);
    
    % Interpolate the joint trajectories over the slow time axis
    motion_radar = zeros(num_pulses, 22, 3);
    for j = 1:22
        % Use piecewise cubic hermite interpolating polynomial (pchip) preserving monotonicity and preventing overshoot avoiding micro-Doppler glitches
        motion_radar(:,j,1) = interp1(t_motion, motion_mat(:,j,1), t_slow, 'pchip', 'extrap'); 
        motion_radar(:,j,2) = interp1(t_motion, motion_mat(:,j,2), t_slow, 'pchip', 'extrap'); 
        motion_radar(:,j,3) = interp1(t_motion, motion_mat(:,j,3), t_slow, 'pchip', 'extrap'); 
    end
    
    % Initialize the raw echo matrix containing all fast and slow time domain samples
    % Preallocate as complex array preventing implicit memory reallocations during pulse injection
    raw_echo = complex(zeros(N_s, num_pulses));
    
    % Generate intrinsic random scattering phases for human skeletal joints to avoid unnatural coherent addition
    rng(42); % Make phases reproducible across simulation runs
    joint_random_phases = 2 * pi * rand(22, 1);
    
    % Compute the boresight direction vectors dynamically pointing roughly towards the center of the room for beam evaluation
    room_center = [0, 0, 1];
    tx_boresight = room_center - tx_pos;
    if norm(tx_boresight) > 1e-3, tx_boresight = tx_boresight / norm(tx_boresight); else, tx_boresight = [0, -1, 0]; end
    rx_boresight = room_center - rx_pos;
    if norm(rx_boresight) > 1e-3, rx_boresight = rx_boresight / norm(rx_boresight); else, rx_boresight = [0, -1, 0]; end

    % Removed the GUI waitbar to maintain the immersive agent console output
    agent_think("Calculating bistatic radar equations and multipath physics across all pulses. This might take some computational effort.");
    for p = 1:num_pulses
        
        % Initialize the signal vector for the current single pulse
        % Preallocate as complex array
        signal_p = complex(zeros(N_s, 1));
        
        % Component 0: Antenna Direct Coupling Leakage Physics
        % Add minimum near-field threshold strictly preventing nonphysical division by absolute zero singularity scaling
        R_leakage = max(norm(tx_pos - rx_pos), 1e-3);
        tau_leakage = R_leakage / c;
        if isfield(Simulation_Params, 'antenna_isolation')
            iso_lin = 10^(-Simulation_Params.antenna_isolation / 20);
        else
            iso_lin = 10^(-40 / 20);
        end
        amp_leakage = (lambda * G_lin) / (4 * pi * R_leakage) * iso_lin;
        phase_leakage = 2*pi*(fc*tau_leakage + K*tau_leakage*t_fast - 0.5*K*tau_leakage^2);
        signal_p = signal_p + amp_leakage .* exp(1i * phase_leakage);

        for j = 1:22
            % Extract the target position array mapping the correct coordinate frames
            target_pos =[motion_radar(p,j,1), motion_radar(p,j,3), motion_radar(p,j,2)];
            
            % Calculate the direct path distance from transmitter to target to receiver
            R_tx = norm(target_pos - tx_pos);
            R_rx = norm(target_pos - rx_pos);
            R_total = R_tx + R_rx;
            
            % Physical Antenna Pattern Factor Calculation
            % Calculate normalized spatial direction vectors modeling realistic beam drop-off preventing isotropic gain artifacts
            v_tx = (target_pos - tx_pos) / (R_tx + 1e-6);
            v_rx = (target_pos - rx_pos) / (R_rx + 1e-6);
            pat_tx = max(dot(v_tx, tx_boresight), 0)^2;
            pat_rx = max(dot(v_rx, rx_boresight), 0)^2;
            pattern_factor = pat_tx * pat_rx;
            
            % Wall Penetration Physical Effect Modeling
            d_tx_wall = 0; tau_tx_wall_delay = 0; trans_loss_tx = 1.0;
            d_rx_wall = 0; tau_rx_wall_delay = 0; trans_loss_rx = 1.0;
            wall_loss_factor = 1.0;
            
            if Simulation_Params.enable_wall
                % Determine signal intersection logic evaluating spatial conditions on Y axis
                if (tx_pos(2) > wall_y_max && target_pos(2) < wall_y_min) || (tx_pos(2) < wall_y_min && target_pos(2) > wall_y_max)
                    cos_theta_tx = abs(target_pos(2) - tx_pos(2)) / (R_tx + 1e-6);
                    d_tx_wall = wall_y_thickness / max(cos_theta_tx, 0.01);
                    tau_tx_wall_delay = d_tx_wall / v_wall - d_tx_wall / c;
                    % Amplitude transmission boundary double sided loss is exactly 1 - Gamma^2
                    trans_loss_tx = max(1 - wall_gamma^2, 0); 
                end
                
                if (rx_pos(2) > wall_y_max && target_pos(2) < wall_y_min) || (rx_pos(2) < wall_y_min && target_pos(2) > wall_y_max)
                    cos_theta_rx = abs(target_pos(2) - rx_pos(2)) / (R_rx + 1e-6);
                    d_rx_wall = wall_y_thickness / max(cos_theta_rx, 0.01);
                    tau_rx_wall_delay = d_rx_wall / v_wall - d_rx_wall / c;
                    % Amplitude transmission boundary double sided loss is exactly 1 - Gamma^2
                    trans_loss_rx = max(1 - wall_gamma^2, 0);
                end
                
                % Exponentially decay the signal magnitude complying with penetration physics
                wall_loss_factor = trans_loss_tx * trans_loss_rx * exp(-wall_alpha * (d_tx_wall + d_rx_wall));
            end
            
            % Calculate the accurate time delay including slowing effect inside the wall
            tau = R_total / c + tau_tx_wall_delay + tau_rx_wall_delay;
            
            % Update spatial amplitude incorporating proper R_tx * R_rx rules maintaining real magnitude scales
            amp = base_amp_factor * sqrt(Simulation_Params.Normalized_RCS(j)) / (R_tx * R_rx + 1e-6) * wall_loss_factor * pattern_factor;
            
            % Calculate the beat signal phase vector and stack the complex exponential wave
            % Embed independent intrinsic scattering center phase preventing uniform artificial interference
            phase = 2*pi*(fc*tau + K*tau*t_fast - 0.5*K*tau^2) + joint_random_phases(j);
            signal_p = signal_p + amp .* exp(1i * phase);
            
            % Simulate the multipath ground bounce effect and complex wall multipath conditionally
            if Simulation_Params.enable_multipath
                % Component 1: Multi-Path Ground Modeling
                % Reduce generic floor reflection coefficient to prevent excessive visual "thickening"
                %   of the main track, allowing the distinct parallel wall ringing ghosts to dominate clearly.
                gamma_gnd = -0.3; % Proper reflection coefficient with phase inversion fading mapping realistically
                
                % Target spatial geometric reflection symmetry across the zero Z horizon
                target_pos_mp = [target_pos(1), target_pos(2), -target_pos(3)];
                R_tx_mp = norm(target_pos_mp - tx_pos);
                R_rx_mp = norm(target_pos_mp - rx_pos);
                
                d_tx_wall_mp = 0; tau_tx_mp_wall_delay = 0; trans_loss_tx_mp = 1.0;
                d_rx_wall_mp = 0; tau_rx_mp_wall_delay = 0; trans_loss_rx_mp = 1.0;
                
                if Simulation_Params.enable_wall
                    if (tx_pos(2) > wall_y_max && target_pos_mp(2) < wall_y_min) || (tx_pos(2) < wall_y_min && target_pos_mp(2) > wall_y_max)
                        cos_theta_tx_mp = abs(target_pos_mp(2) - tx_pos(2)) / (R_tx_mp + 1e-6);
                        d_tx_wall_mp = wall_y_thickness / max(cos_theta_tx_mp, 0.01);
                        tau_tx_mp_wall_delay = d_tx_wall_mp / v_wall - d_tx_wall_mp / c;
                        trans_loss_tx_mp = max(1 - wall_gamma^2, 0);
                    end
                    if (rx_pos(2) > wall_y_max && target_pos_mp(2) < wall_y_min) || (rx_pos(2) < wall_y_min && target_pos_mp(2) > wall_y_max)
                        cos_theta_rx_mp = abs(target_pos_mp(2) - rx_pos(2)) / (R_rx_mp + 1e-6);
                        d_rx_wall_mp = wall_y_thickness / max(cos_theta_rx_mp, 0.01);
                        tau_rx_mp_wall_delay = d_rx_wall_mp / v_wall - d_rx_wall_mp / c;
                        trans_loss_rx_mp = max(1 - wall_gamma^2, 0);
                    end
                end
                
                % Route 1.1: TX -> Ground -> Target -> RX
                v_tx_mpA = (target_pos_mp - tx_pos) / (R_tx_mp + 1e-6);
                pat_tx_mpA = max(dot(v_tx_mpA, tx_boresight), 0)^2;
                pattern_factor_A = pat_tx_mpA * pat_rx;
                
                tau_mpA = (R_tx_mp + R_rx) / c + tau_tx_mp_wall_delay + tau_rx_wall_delay;
                amp_loss_A = trans_loss_tx_mp * trans_loss_rx * exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall));
                amp_mpA = gamma_gnd * base_amp_factor * sqrt(Simulation_Params.Normalized_RCS(j)) / (R_tx_mp * R_rx + 1e-6) * amp_loss_A * pattern_factor_A; 
                phase_mpA = 2*pi*(fc*tau_mpA + K*tau_mpA*t_fast - 0.5*K*tau_mpA^2) + joint_random_phases(j);
                signal_p = signal_p + amp_mpA .* exp(1i * phase_mpA);
                
                % Route 1.2: TX -> Target -> Ground -> RX
                v_rx_mpB = (target_pos_mp - rx_pos) / (R_rx_mp + 1e-6);
                pat_rx_mpB = max(dot(v_rx_mpB, rx_boresight), 0)^2;
                pattern_factor_B = pat_tx * pat_rx_mpB;
                
                tau_mpB = (R_tx + R_rx_mp) / c + tau_tx_wall_delay + tau_rx_mp_wall_delay;
                amp_loss_B = trans_loss_tx * trans_loss_rx_mp * exp(-wall_alpha * (d_tx_wall + d_rx_wall_mp));
                amp_mpB = gamma_gnd * base_amp_factor * sqrt(Simulation_Params.Normalized_RCS(j)) / (R_tx * R_rx_mp + 1e-6) * amp_loss_B * pattern_factor_B; 
                phase_mpB = 2*pi*(fc*tau_mpB + K*tau_mpB*t_fast - 0.5*K*tau_mpB^2) + joint_random_phases(j);
                signal_p = signal_p + amp_mpB .* exp(1i * phase_mpB);
                
                % Route 1.3: TX -> Ground -> Target -> Ground -> RX
                pattern_factor_C = pat_tx_mpA * pat_rx_mpB;
                
                tau_mpC = (R_tx_mp + R_rx_mp) / c + tau_tx_mp_wall_delay + tau_rx_mp_wall_delay;
                amp_loss_C = trans_loss_tx_mp * trans_loss_rx_mp * exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall_mp));
                amp_mpC = (gamma_gnd^2) * base_amp_factor * sqrt(Simulation_Params.Normalized_RCS(j)) / (R_tx_mp * R_rx_mp + 1e-6) * amp_loss_C * pattern_factor_C; 
                phase_mpC = 2*pi*(fc*tau_mpC + K*tau_mpC*t_fast - 0.5*K*tau_mpC^2) + joint_random_phases(j);
                signal_p = signal_p + amp_mpC .* exp(1i * phase_mpC);
                
                % Component 2, 3 & 4: Wall Structural Multipath Signatures
                if Simulation_Params.enable_wall
                    % Determine the active wall face tracking bouncing reverberations
                    if target_pos(2) > wall_y_max
                        target_wall_face = wall_y_max;
                    else
                        target_wall_face = wall_y_min;
                    end
                    
                    % Component 2: Exterior bounce modeling utilizing strict Bistatic Image Theory Geometry
                    target_pos_wall_mp = target_pos;
                    target_pos_wall_mp(2) = 2 * target_wall_face - target_pos(2);
                    R_tx_wall_mp = norm(target_pos_wall_mp - tx_pos);
                    R_rx_wall_mp = norm(target_pos_wall_mp - rx_pos);
                    
                    % Path 2A: TX -> Wall -> Target -> RX (Valid iff TX and Target stay on the exact same side of the wall surface)
                    if (tx_pos(2) - target_wall_face) * (target_pos(2) - target_wall_face) > 0
                        v_tx_wall_mp = (target_pos_wall_mp - tx_pos) / (R_tx_wall_mp + 1e-6);
                        pat_tx_wall_mp = max(dot(v_tx_wall_mp, tx_boresight), 0)^2;
                        pattern_factor_2A = pat_tx_wall_mp * pat_rx;
                        
                        tau_mp2A = (R_tx_wall_mp + R_rx) / c + tau_rx_wall_delay; 
                        amp_mp2A = wall_gamma * base_amp_factor * sqrt(Simulation_Params.Normalized_RCS(j)) / (R_tx_wall_mp * R_rx + 1e-6) * trans_loss_rx * exp(-wall_alpha * d_rx_wall) * pattern_factor_2A;
                        phase_mp2A = 2*pi*(fc*tau_mp2A + K*tau_mp2A*t_fast - 0.5*K*tau_mp2A^2) + joint_random_phases(j);
                        signal_p = signal_p + amp_mp2A .* exp(1i * phase_mp2A);
                    end
                    
                    % Path 2B: TX -> Target -> Wall -> RX (Valid iff RX and Target stay on the exact same side of the wall surface)
                    if (rx_pos(2) - target_wall_face) * (target_pos(2) - target_wall_face) > 0
                        v_rx_wall_mp = (target_pos_wall_mp - rx_pos) / (R_rx_wall_mp + 1e-6);
                        pat_rx_wall_mp = max(dot(v_rx_wall_mp, rx_boresight), 0)^2;
                        pattern_factor_2B = pat_tx * pat_rx_wall_mp;
                        
                        tau_mp2B = (R_tx + R_rx_wall_mp) / c + tau_tx_wall_delay;
                        amp_mp2B = wall_gamma * base_amp_factor * sqrt(Simulation_Params.Normalized_RCS(j)) / (R_tx * R_rx_wall_mp + 1e-6) * trans_loss_tx * exp(-wall_alpha * d_tx_wall) * pattern_factor_2B;
                        phase_mp2B = 2*pi*(fc*tau_mp2B + K*tau_mp2B*t_fast - 0.5*K*tau_mp2B^2) + joint_random_phases(j);
                        signal_p = signal_p + amp_mp2B .* exp(1i * phase_mp2B);
                    end
                    
                    % Component 3: Upgraded to Multi-Order Internal Wall Reverberation.
                    % Real through-wall RTMs exhibit distinct parallel ghost tracks.
                    % This is physically caused by the radar wave bouncing back-and-forth multiple times 
                    % inside the wall structure before reaching the target or returning to the receiver.
                    if (d_tx_wall > 0 || d_rx_wall > 0)
                        max_ringing_order = 5; % Generate up to 5 parallel multipath ghost tracks
                        
                        % Real walls have multiple internal interfaces, leading to stronger 
                        % internal reflections than a perfectly homogeneous solid block.
                        % We phenomenologically enhance the effective round-trip reflection coefficient.
                        effective_gamma_rt = min(abs(wall_gamma) * 2.0, 0.85); 
                        
                        for m_ring = 1:max_ringing_order
                            d_internal_bounce = m_ring * 2 * wall_y_thickness;
                            tau_mp3 = tau + d_internal_bounce / v_wall;
                            
                            % Attenuate with internal path exponent tracking ohmic propagation absorption explicitly.
                            % Note: 'amp' already contains pattern_factor and base physical transmission losses.
                            amp_mp3 = amp * (effective_gamma_rt^m_ring) * exp(-wall_alpha * d_internal_bounce); 
                            
                            phase_mp3 = 2*pi*(fc*tau_mp3 + K*tau_mp3*t_fast - 0.5*K*tau_mp3^2) + joint_random_phases(j);
                            signal_p = signal_p + amp_mp3 .* exp(1i * phase_mp3);
                        end
                    end
                    
                    % Component 4: Target-Wall Room Reverberation
                    % Simulates the wave reflecting off the target, hitting the indoor wall face, 
                    % and bouncing back to the target again. Creates delayed tracks with steeper dynamic slopes (V-shapes).
                    if target_pos(2) < wall_y_min && tx_pos(2) > wall_y_max
                        inner_wall_face = wall_y_min;
                        is_valid_twt = true;
                    elseif target_pos(2) > wall_y_max && tx_pos(2) < wall_y_min
                        inner_wall_face = wall_y_max;
                        is_valid_twt = true;
                    else
                        is_valid_twt = false; % Target is inside the wall, theoretically impossible but handled safely
                    end
                    
                    if is_valid_twt
                        D_tw = abs(target_pos(2) - inner_wall_face);
                        if D_tw > 0.1
                            % Path 4A: 1st-Order Reverberation (Target -> Wall -> Target)
                            tau_mp4 = tau + 2 * D_tw / c;
                            % Apply physical geometric divergence decay for the extra spatial bounce path.
                            % 'amp' is reused directly without re-applying antenna pattern bounds.
                            amp_mp4 = amp * abs(wall_gamma) * (R_total / (R_total + 2 * D_tw)) * 0.5;
                            phase_mp4 = 2*pi*(fc*tau_mp4 + K*tau_mp4*t_fast - 0.5*K*tau_mp4^2) + joint_random_phases(j);
                            signal_p = signal_p + amp_mp4 .* exp(1i * phase_mp4);
                            
                            % Path 4B: 2nd-Order Reverberation (Target -> Wall -> Target -> Wall -> Target)
                            tau_mp5 = tau + 4 * D_tw / c;
                            amp_mp5 = amp * (abs(wall_gamma)^2) * (R_total / (R_total + 4 * D_tw)) * 0.25;
                            phase_mp5 = 2*pi*(fc*tau_mp5 + K*tau_mp5*t_fast - 0.5*K*tau_mp5^2) + joint_random_phases(j);
                            signal_p = signal_p + amp_mp5 .* exp(1i * phase_mp5);
                        end
                    end
                end
            end
        end
        % Assign the synthesized pulse to the entire raw echo matrix
        raw_echo(:, p) = signal_p;
    end
    
    % Accommodate the FFT processing gain over N_s enabling rigorous physics level SNR simulations rendering visually consistent noise floor
    ref_distance = 2.0;
    ref_amp = (lambda * G_lin * 1.0) / ((4*pi)^1.5 * ref_distance^2);
    ref_power = ref_amp^2;
    noise_power = (ref_power / (10^(Simulation_Params.SNR / 10))) * N_s;
    noise_matrix = sqrt(noise_power/2) * (randn(size(raw_echo)) + 1i * randn(size(raw_echo)));
    
    raw_echo = raw_echo + noise_matrix;
    save(fullfile(Save_Dir, 'Raw_Echo.mat'), 'raw_echo', '-v7.3');
    
    %% Signal Processing: RTM & DTM Generation
    % Replace default print with agent thinking stream
    agent_think("Executing discrete numerical fourier transforms upon the simulated echoes to map electromagnetic propagation properties into spatial range and velocity doppler coordinates.");
    
    % Apply structural windowing function critically reducing fast time sinc range sidelobes
    fast_time_window = hamming(N_s);
    raw_echo_windowed = raw_echo .* fast_time_window;
    
    % Apply fast fourier transform along the fast time dimension
    range_fft = fft(raw_echo_windowed, N_s, 1);
    
    % Keep only the positive frequencies to construct the complex range time matrix
    RTM_complex = range_fft(1:floor(N_s/2), :);         
    
    % Calculate the corresponding physical range axis mapped to the beat frequencies
    f_beat = (0:floor(N_s/2)-1)' * (fs / N_s);
    range_axis = f_beat * c / (2 * K);

    % Define a realistic maximum boundary for the indoor physical environment
    % Extended to 25.0 meters accommodating distant high-order through-wall multipath ghost trajectories
    max_indoor_range = 25.0; 
    
    % Filter out the redundant distant range bins accelerating neural network inference
    valid_range_idx = range_axis <= max_indoor_range;
    RTM_complex = RTM_complex(valid_range_idx, :);
    range_axis = range_axis(valid_range_idx);
    
    % Clear immense memory blocks releasing gigabytes of system resources ensuring simulation stability preventing OOM
    clear range_fft raw_echo_windowed raw_echo noise_matrix;
    
    % Apply Moving Target Indication (MTI) subtracting static constant targets (such as walls and DC leakage component)
    RTM_complex = RTM_complex - mean(RTM_complex, 2);
    
    % Embed hardware Sensitivity Time Control (STC) overcoming long range dramatic amplitude collapsing behaviors via R^2 weight
    STC_Weights = (range_axis.^2) + 1e-3;
    [~, ref_idx] = min(abs(range_axis - 2.0));
    STC_Weights = STC_Weights / STC_Weights(max(1, ref_idx)); 
    RTM_complex = RTM_complex .* STC_Weights;

    % Compute the linear forms of the range time magnitude array
    RTM_Mag_Lin = abs(RTM_complex);
    
    % Dynamic continuous tracking capturing unconstrained motions bounding valid spatial locations perfectly evading signal truncations
    range_variance = var(RTM_Mag_Lin, 0, 2);
    active_threshold = max(range_variance) * 0.05; 
    active_bins = find(range_variance > active_threshold);
    
    if isempty(active_bins)
        [~, max_idx] = max(mean(RTM_Mag_Lin, 2));
        active_bins = max_idx;
    end
    
    % Establish dynamic plotting bounds dynamically framing actions fully resolving "walking circles"
    % Extend the upper bound significantly to ensure all far-range multipath ghosts are clearly visualized
    plot_r_min = max(0, range_axis(active_bins(1)) - 2.0);
    plot_r_max = min(max_indoor_range, range_axis(active_bins(end)) + 8.0);
    
    % Preallocate time and frequency parameters extracting proper dimensions via a single spectrogram call
    n_window = max(round(Simulation_Params.stft_window_seconds * PRF), 8);
    n_overlap = min(round(n_window * Simulation_Params.stft_overlap_ratio), n_window - 1);
    [~, F_doppler, time_stft] = spectrogram(RTM_complex(active_bins(1), :), hamming(n_window), n_overlap, 1024, PRF, 'centered');
    
    % Incoherent STFT Integration: Summing magnitude-squared energy globally circumventing structural truncation
    DTM_Energy = zeros(length(F_doppler), length(time_stft));
    for b = 1:length(active_bins)
        bin_idx = active_bins(b);
        [S_bin, ~, ~] = spectrogram(RTM_complex(bin_idx, :), hamming(n_window), n_overlap, 1024, PRF, 'centered');
        DTM_Energy = DTM_Energy + abs(S_bin).^2;
    end
    
    % Re-polarize visual direction aligning target approach behavior conforming to common strictly positive Doppler expectations 
    DTM_Energy = flipud(DTM_Energy);
    DTM_Mag_Lin = sqrt(DTM_Energy);

    %% Magnitude Normalization and Logarithmic Conversion
    % Replace default print with agent thinking stream
    agent_think("Normalizing matrix magnitudes and converting to logarithmic decibel scales to properly visualize the dynamic range of the radar signatures.");
    
    % Compute standard decibel (dB) logarithmic scaling correctly applying 20log10 before offset bounds mapping directly
    RTM_Mag_Log = 20*log10(RTM_Mag_Lin + 1e-12);
    DTM_Mag_Log = 20*log10(DTM_Mag_Lin + 1e-12);
    
    % Align the upper global ceiling strictly mapping peak responses mapping to pristine 0 dB
    RTM_Mag_Log_Norm = RTM_Mag_Log - max(RTM_Mag_Log(:));
    DTM_Mag_Log_Norm = DTM_Mag_Log - max(DTM_Mag_Log(:));
    
    % Re-Normalize mapping raw linear boundaries primarily for generic plotting visual implementations
    RTM_Mag_Lin_Norm = (RTM_Mag_Lin - min(RTM_Mag_Lin(:))) / (max(RTM_Mag_Lin(:)) - min(RTM_Mag_Lin(:)) + 1e-12);
    DTM_Mag_Lin_Norm = (DTM_Mag_Lin - min(DTM_Mag_Lin(:))) / (max(DTM_Mag_Lin(:)) - min(DTM_Mag_Lin(:)) + 1e-12);

    %% Network Denoising
    if Simulation_Params.enable_network
        % Replace default print with agent thinking stream
        agent_think("Connecting the local deep learning denoising framework to systematically eliminate background static while preserving dynamic kinematic traces.");
        try
            % Load the predefined image denoising convolutional neural network
            net = denoisingNetwork('dncnn');
            
            % Cast to single protecting computational memory securing reliable denoise inference executions
            denoise_matrix = @(Mat) ...
                double(denoiseImage(single((Mat - min(Mat(:))) / (max(Mat(:)) - min(Mat(:)) + 1e-12)), net)) * (max(Mat(:)) - min(Mat(:))) + min(Mat(:));
            
            % Execute the noise removal logic utilizing robust scaled definitions enhancing the physical shapes
            RTM_Mag_Log_Norm = denoise_matrix(RTM_Mag_Log_Norm);
            DTM_Mag_Log_Norm = denoise_matrix(DTM_Mag_Log_Norm);
            
            RTM_Mag_Lin_Norm = denoise_matrix(RTM_Mag_Lin_Norm);
            DTM_Mag_Lin_Norm = denoise_matrix(DTM_Mag_Lin_Norm);
            
        catch ME
            % Replace default warning with agent thinking stream
            agent_think("The deep neural network enhancement mechanism produced an internal operational failure. Terminating the refinement routine to maintain spectral stability.");
        end
    end

    %% Export High Resolution Dataset Images
    % Replace default print with agent thinking stream
    agent_think("Formulating compressed high fidelity image matrices and writing them to the allocated drive space for downstream neural network integration.");
    
    % Define the fixed logarithmic dynamic range limits for feature enhancement
    log_clim_min = -30;
    log_clim_max = 0;
    
    % Clamp the normalized logarithmic matrices enforcing the specified visual limits
    RTM_Mag_Log_Clamped = max(min(RTM_Mag_Log_Norm, log_clim_max), log_clim_min);
    DTM_Mag_Log_Clamped = max(min(DTM_Mag_Log_Norm, log_clim_max), log_clim_min);
    
    % Map the custom colormap dynamically replacing generic jet mappings maintaining visual fidelity
    custom_cmap_256 = interp1(linspace(0, 1, size(JoeyBG_Colormap_Flip, 1)), JoeyBG_Colormap_Flip, linspace(0, 1, 256));
    
    % Normalize and export the full range logarithmic range time image
    rtm_norm_img = (RTM_Mag_Log_Clamped - log_clim_min) ./ (log_clim_max - log_clim_min);
    rtm_rgb = ind2rgb(gray2ind(rtm_norm_img, 256), custom_cmap_256);
    % Compress exorbitant sizes to generic memory-friendly scale maintaining rigorous machine learning shapes
    rtm_resized = imresize(rtm_rgb, [1024, 1024]);
    imwrite(rtm_resized, fullfile(Save_Dir, 'RTM_Log_1024x1024_Image.jpg'));
    
    % Normalize and export the full range logarithmic doppler time image
    dtm_norm_img = (DTM_Mag_Log_Clamped - log_clim_min) ./ (log_clim_max - log_clim_min);
    dtm_rgb = ind2rgb(gray2ind(dtm_norm_img, 256), custom_cmap_256);
    % Compress exorbitant sizes to generic memory-friendly scale maintaining rigorous machine learning shapes
    dtm_resized = imresize(dtm_rgb,[1024, 1024]);
    imwrite(dtm_resized, fullfile(Save_Dir, 'DTM_Log_1024x1024_Image.jpg'));

    %% Plotting RTM and DTM User Interfaces
    % Replace default print with agent thinking stream
    agent_think("Generating independent graphical interface components displaying the structural distributions of the simulated matrices for visual inspection.");
    
    % Render the user interface figure for the linear range time matrix
    fig_rtm_lin = figure('Name', 'Range-Time Matrix (Linear)', 'Position',[100, 100, 700, 500]);
    ax_rtm_lin = axes('Parent', fig_rtm_lin);
    imagesc(ax_rtm_lin, t_slow, range_axis, RTM_Mag_Lin_Norm);
    set(ax_rtm_lin, 'YDir', 'normal', 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
    colormap(ax_rtm_lin, JoeyBG_Colormap_Flip); colorbar(ax_rtm_lin);
    % Apply defined color limits highlighting structural features overriding defaults
    clim(ax_rtm_lin, [0, 1]);
    title('RTM', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
    xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylim(ax_rtm_lin,[plot_r_min, plot_r_max]); 
    
    % Render the user interface figure for the linear doppler time matrix
    fig_dtm_lin = figure('Name', 'Doppler-Time Matrix (Linear)', 'Position',[150, 150, 700, 500]);
    ax_dtm_lin = axes('Parent', fig_dtm_lin);
    imagesc(ax_dtm_lin, time_stft, F_doppler, DTM_Mag_Lin_Norm);
    set(ax_dtm_lin, 'YDir', 'normal', 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
    colormap(ax_dtm_lin, JoeyBG_Colormap_Flip); colorbar(ax_dtm_lin);
    % Apply defined color limits highlighting structural features overriding defaults
    clim(ax_dtm_lin, [0, 1]);
    title('DTM', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
    xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylim(ax_dtm_lin, [-PRF/2, PRF/2]); 
    
    % Render the user interface figure for the logarithmic range time matrix
    fig_rtm_log = figure('Name', 'Range-Time Matrix (Log)', 'Position',[200, 200, 700, 500]);
    ax_rtm_log = axes('Parent', fig_rtm_log);
    imagesc(ax_rtm_log, t_slow, range_axis, RTM_Mag_Log_Norm);
    set(ax_rtm_log, 'YDir', 'normal', 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
    colormap(ax_rtm_log, JoeyBG_Colormap_Flip); colorbar(ax_rtm_log);
    % Apply defined color limits highlighting structural features overriding defaults
    clim(ax_rtm_log, [-30, 0]);
    title('RTM (Log)', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
    xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylim(ax_rtm_log,[plot_r_min, plot_r_max]); 
    
    % Render the user interface figure for the logarithmic doppler time matrix
    fig_dtm_log = figure('Name', 'Doppler-Time Matrix (Log)', 'Position',[250, 250, 700, 500]);
    ax_dtm_log = axes('Parent', fig_dtm_log);
    imagesc(ax_dtm_log, time_stft, F_doppler, DTM_Mag_Log_Norm);
    set(ax_dtm_log, 'YDir', 'normal', 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
    colormap(ax_dtm_log, JoeyBG_Colormap_Flip); colorbar(ax_dtm_log);
    % Apply defined color limits highlighting structural features overriding defaults
    clim(ax_dtm_log, [-30, 0]);
    title('DTM (Log)', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
    xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylim(ax_dtm_log, [-PRF/2, PRF/2]); 

    %% Visualizing the Motion
    % Replace default print with agent thinking stream
    agent_think("Constructing the primary three dimensional workspace rendering pipeline to project and animate the physical motions and structural scene layouts concurrently.");
    
    frames = Simulation_Params.num_frames;

    % Setup the main figure and graphical axes using given formatting arrays
    fig = figure('Name', 'EMDM 3D Motion Visualization', 'Position',[100, 100, 900, 800]);
    ax = axes('Parent', fig);
    hold(ax, 'on');
    grid(ax, 'on');
    
    % Apply the provided visualization font styles and title labels
    set(ax, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
    title('Motion Visualization', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
    xlabel('X (Azimuth, m)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylabel('Y (Range, m)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    zlabel('Z (Height, m)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    
    % Set the viewing angle to observe the target appropriately
    view(ax, 45, 20);
    
    % Bind the colormap mapping correctly for RCS visualization
    colormap(ax, JoeyBG_Colormap_Flip);
    clim(ax,[0, 1]); 
    
    % Add the colorbar mapping bar to the edge of the visualizer
    cb = colorbar(ax);
    ylabel(cb, 'Normalized RCS Value', 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
    
    % Calculate the dynamic spatial boundaries for the three dimensional scene
    X_all_frames = motion_mat(:, :, 1);
    Y_all_frames = motion_mat(:, :, 3); 
    
    pad = 0.5;
    x_min = min(min(X_all_frames,[], 'all'), min([tx_pos(1), rx_pos(1)])) - pad;
    x_max = max(max(X_all_frames,[], 'all'), max([tx_pos(1), rx_pos(1)])) + pad;
    y_min = min(min(Y_all_frames,[], 'all'), min([tx_pos(2), rx_pos(2)])) - pad;
    y_max = max(max(Y_all_frames,[], 'all'), max([tx_pos(2), rx_pos(2)])) + pad;
    
    xlim(ax,[x_min, x_max]);
    ylim(ax,[y_min, y_max]);
    zlim(ax,[0, max(tx_pos(3), 2.2)]);
    
    % Generate the surface grid domain spanning the ground plane area
    grid_step = 0.15; 
    x_grid_vals = floor(x_min):grid_step:ceil(x_max);
    y_grid_vals = floor(y_min):grid_step:ceil(y_max);
    [GX, GY] = meshgrid(x_grid_vals, y_grid_vals);
    GZ = zeros(size(GX));
    
    % Prevent delaunay algorithm crashing on regular structured meshes directly utilizing surface grids
    surf(GX, GY, GZ, 'Parent', ax, ...
         'FaceColor',[0.25 0.25 0.25], 'FaceAlpha', 0.6, ...
         'EdgeColor', [0.15 0.15 0.15], 'EdgeAlpha', 0.5, ...
         'HandleVisibility', 'off');
         
    % Plot the transmitting and receiving antenna markers in the space
    plot3(ax, tx_pos(1), tx_pos(2), tx_pos(3), '^', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(2,:), 'MarkerEdgeColor', 'k', 'DisplayName', 'TX Antenna');
    plot3(ax, rx_pos(1), rx_pos(2), rx_pos(3), 'v', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(end-1,:), 'MarkerEdgeColor', 'k', 'DisplayName', 'RX Antenna');
    
    % Plot the semi transparent simulation wall cube if the flag is enabled
    if Simulation_Params.enable_wall
        wc = Simulation_Params.wall_center;
        wd = Simulation_Params.wall_dimensions;
        
        X_w = wc(1) + [-1 1 1 -1 -1 1 1 -1] * wd(1)/2;
        Y_w = wc(2) +[-1 -1 1 1 -1 -1 1 1] * wd(2)/2;
        Z_w = wc(3) +[-1 -1 -1 -1 1 1 1 1] * wd(3)/2;
        
        faces =[1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
        patch('Vertices',[X_w' Y_w' Z_w'], 'Faces', faces, 'Parent', ax, ...
              'FaceColor',[0.7 0.7 0.8], 'FaceAlpha', 0.3, 'EdgeColor', [0.4 0.4 0.5], ...
              'DisplayName', 'Simulation Wall');
    end
          
    % Add lighting effects and material reflections to enhance visual realism
    camlight(ax, 'headlight');
    camlight(ax, 'left');
    lighting(ax, 'gouraud');
    material(ax, 'dull'); 
    
    % Extract colors to display the categorical properties in the primary legend
    rep_joint_color = JoeyBG_Colormap(4, :);
    rep_bone_color = JoeyBG_Colormap_Flip(4, :);
    
    plot3(ax, nan, nan, nan, 'o', 'MarkerFaceColor', rep_joint_color, 'MarkerEdgeColor', 'none', 'MarkerSize', 10, 'DisplayName', 'Human Joints');
    plot3(ax, nan, nan, nan, '-', 'Color', rep_bone_color, 'LineWidth', 4, 'DisplayName', 'Human Bones');
    plot3(ax, nan, nan, nan, '-', 'Color', [rep_joint_color, 0.4], 'LineWidth', 1.5, 'DisplayName', 'Joint Trajectories');
    legend(ax, 'Location', 'northeast', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);

    % Preallocate three dimensional volumetric primitive templates
    [cyl_x, cyl_y, cyl_z] = cylinder([1.0, 0.65], 20); 
    [sph_x, sph_y, sph_z] = sphere(20);                
    
    % Parse the dimension scalar metrics from the parameters collection
    num_joints = Simulation_Params.num_joints;
    num_bones = size(kinematic_tree, 1);
    cmap_len = size(JoeyBG_Colormap, 1);
    
    % Round mapping boundaries safely mapping the colors to radar section indices
    rcs_color_indices = round((1 - Normalized_RCS) * (cmap_len - 1)) + 1;
    rcs_color_indices = max(min(rcs_color_indices, cmap_len), 1); 
    
    % Create graphical object array structures to handle massive graphics updating
    h_joints_t = gobjects(num_joints, 1);
    h_bones_t = gobjects(num_bones, 1);
    h_trajs = gobjects(num_joints, 1);
    
    % Initialize the three dimensional spherical joints and hidden trailing tracks
    for j = 1:num_joints
        rcs_idx = rcs_color_indices(j);
        joint_color = JoeyBG_Colormap(rcs_idx, :);
        
        h_joints_t(j) = hgtransform('Parent', ax);
        surf(sph_x, sph_y, sph_z, 'Parent', h_joints_t(j), 'FaceColor', joint_color, ...
             'EdgeColor', 'none', 'HandleVisibility', 'off', ...
             'SpecularStrength', 0.2, 'DiffuseStrength', 0.8);
             
        h_trajs(j) = plot3(ax, nan, nan, nan, 'Color',[joint_color, 0.35], ...
                           'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
    
    % Initialize the three dimensional cylindrical bones blending connected colors
    for b = 1:num_bones
        j1 = kinematic_tree(b, 1);
        j2 = kinematic_tree(b, 2);
        
        color1 = JoeyBG_Colormap(rcs_color_indices(j1), :);
        color2 = JoeyBG_Colormap(rcs_color_indices(j2), :);
        
        num_points = size(cyl_z, 2);
        bone_cdata = zeros(2, num_points, 3);
        for c_idx = 1:3
            bone_cdata(1, :, c_idx) = color1(c_idx);
            bone_cdata(2, :, c_idx) = color2(c_idx);
        end
        
        h_bones_t(b) = hgtransform('Parent', ax);
        
        surf(cyl_x, cyl_y, cyl_z, bone_cdata, 'Parent', h_bones_t(b), ...
             'FaceColor', 'interp', ...
             'EdgeColor', 'none', 'HandleVisibility', 'off', ...
             'SpecularStrength', 0.1, 'DiffuseStrength', 0.9);
    end
    
    % Allocate NaN arrays tracking spatial trailing histories frame iteratively
    traj_X = nan(frames, num_joints);
    traj_Y = nan(frames, num_joints);
    traj_Z = nan(frames, num_joints);

    % Removed GUI waitbar for 3D visualization and replaced with thinking stream
    agent_think("Synchronizing internal plot rendering loops to animate the target joints along their mathematically resolved spatial vectors in real time.");
    
    % Execute the primary animation loop over all chronological sample frames
    for f = 1:frames
        if ~isgraphics(ax)
            break; 
        end
        
        % Extract the three dimensional coordinates slicing the entire matrix block
        curr_joints = squeeze(motion_mat(f, :, :));
        
        % Remap the standard axis vectors replacing coordinates natively 
        X_data = curr_joints(:, 1);
        Y_data = curr_joints(:, 3); 
        Z_data = curr_joints(:, 2); 
        
        % Record spatial trajectories continually appending location variables
        traj_X(f, :) = X_data';
        traj_Y(f, :) = Y_data';
        traj_Z(f, :) = Z_data';
        
        % Update the graphic spatial trailing instances drawing connected polylines
        for j = 1:num_joints
            h_trajs(j).XData = traj_X(1:f, j);
            h_trajs(j).YData = traj_Y(1:f, j);
            h_trajs(j).ZData = traj_Z(1:f, j);
        end
        
        % Manipulate the spherical transformation blocks updating respective centers scaling
        for j = 1:num_joints
            T = makehgtform('translate',[X_data(j), Y_data(j), Z_data(j)]);
            S = makehgtform('scale', Simulation_Params.joint_radii(j));
            h_joints_t(j).Matrix = T * S;
        end
        
        % Reconstruct the intermediate bone transformation blocks updating lengths rotation
        for b = 1:num_bones
            j1 = kinematic_tree(b, 1);
            j2 = kinematic_tree(b, 2);
            
            p1 =[X_data(j1), Y_data(j1), Z_data(j1)];
            p2 =[X_data(j2), Y_data(j2), Z_data(j2)];
            
            h_bones_t(b).Matrix = compute_bone_matrix(p1, p2, Simulation_Params.bone_radius);
        end
        
        drawnow;
        pause(1/Simulation_Params.FPS); 
    end
    
    % Replace default print with agent thinking stream
    agent_think("The entire complex sequence of mathematical simulation formulations and graphical layout processes has been successfully concluded.");

catch ME
    % Waitbars removed completely, replacing with direct stream and stack trace outputs
    agent_think("A severe procedural violation occurred abruptly halting the algorithmic flow. Rolling back to original directory context and printing the system exception memory trace below for physical debugging.");
    
    if exist('original_dir', 'var')
        cd(original_dir);
    end
    rethrow(ME);
end

%% Local Functions
% Added local routine to simulate an agent sequentially streaming cognitive thoughts into the command window
function agent_think(text_str)
    % prefix = 'Thinking: ';
    % fprintf('\n%s', prefix);
    char_array = char(text_str);
    for idx = 1:length(char_array)
        fprintf('%c', char_array(idx));
        pause(0.015);
    end
    fprintf('\n');
end

% Define the sub routine to evaluate generic mapping translation rotation elements connecting extremities
function M = compute_bone_matrix(p1, p2, radius)
    v = p2 - p1;
    L = norm(v);
    
    % Mask the invalid geometries nulling scaling magnitude explicitly
    if L < 1e-5
        M = makehgtform('scale', 0);
        return;
    end
    
    % Evaluate the orienting cosine products estimating target angles properly
    dir_vec = v / L;
    z_axis =[0, 0, 1];
    
    % Construct rotational instances differentiating co linear singular vectors cautiously
    if norm(cross(z_axis, dir_vec)) < 1e-5
        if dot(z_axis, dir_vec) > 0
            R = eye(4); 
        else
            R = makehgtform('xrotate', pi); 
        end
    else
        rot_axis = cross(z_axis, dir_vec);
        rot_axis = rot_axis / norm(rot_axis);
        rot_angle = acos(dot(z_axis, dir_vec));
        R = makehgtform('axisrotate', rot_axis, rot_angle);
    end
    
    % Stack the final concatenated sequential block defining spatial states accurately
    T = makehgtform('translate', p1);
    S = makehgtform('scale',[radius, radius, L]);
    
    M = T * R * S;
end