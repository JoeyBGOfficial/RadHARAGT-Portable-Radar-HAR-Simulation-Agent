%% RadHARAGT Main Script
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2026-04-20.
% Affiliate: Beijing Institute of Technology.
% Platform: Ollama, MATLAB R2025b, Python 3.10.13 with Conda EMDM Environment.
%
% Introduction:
%   This script implements a comprehensive pipeline for generating realistic indoor multi-human radar echo datasets.
%   It utilizes a large language model via a local REST API to interpret user descriptions and extract precise radar simulation parameters,
%   including individual starting positions, orientations, delays, physical sizes for multiple targets, and discrete scattering point clouds for static objects.
%   The script refines the input into optimized prompts to drive a python-based motion diffusion model environment iteratively.
%   It intercepts newly generated human motion sequences, applies spatial-temporal transformations, and scales RCS based on height and weight.
%   The motion sequences undergo signal post-processing including Savitzky-Golay filtering and variance-based idle frame truncation.
%   The script evaluates spatial intersections dynamically enforcing collision avoidance models tracking unified temporal physical bounds and static structural objects.
%   The script simulates frequency modulated continuous wave radar point-scatterer echoes based on the three-dimensional trajectories, static scene scatterers, and physical radar equations.
%   It fundamentally supports Multi-Input Multi-Output (MIMO) architectures extracting multi-channel raw traces accurately over arbitrary 3D arrays.
%   The system calculates the Range-Time Matrix and Doppler-Time Matrix using fast Fourier transforms extracting isolated First Channel and Channel Sum features.
%   It optionally employs a pre-trained deep convolutional neural network to reduce background noise from the spectral representations.
%   The script exports high-resolution logarithmic magnitude images tailored for training deep learning models.
%   Finally, it renders an animated three-dimensional visualization of the human skeleton motions alongside structural environment objects within the simulated physical scene.
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
%   Ollama local deployment configured with gemma4:e4b model suggested.

%% Preparation for Matlab Script
close all;
clear all;
clc;

% Replace default print with agent thinking stream
agent_think("Author: JoeyBG. RadHARAGT Multi-Person, Object Boundary & MIMO system initialized.");

%% Definition
% Define the basic parameters for the simulation
Model_Name = "gemma4:e4b";                                                  % Name of the LLM model used for prompt refining and information extracting                                                                         
Input_Text = "我现在需要仿真两个人的行为。第一个人身高1.83米，体重79公斤，从坐标[-1, 1.5, 0]处起始，朝向60度，无延迟，保持中速走路。第二个人身高1.62米，体重52公斤，从坐标[2, 2.5, 0]处起始，朝向210度，等待1.5秒后开始小跑。场景中在[1.5, 2.2, 0]处放置一张长1.2米、宽0.8米、高0.75米的桌子。采用中心频率2GHz、带宽1GHz的雷达，PRF为200Hz，发射天线位置为[0, 5, 1]，接收天线位置为[0.2, 5, 1]、[0.4, 5, 1]和[0.6, 5, 1]，天线增益15dBi，系统信噪比45dB。启用墙体，墙体位置设置为[0, 4, 1]，墙体宽度8米、高度3米、厚度0.12米，介电常数4.5，损耗角正切0.05。需要考虑多径效应，不使用神经网络进行特征增强。";
Has_Image = false;                                                          % Define whether the LLM use image for input                                                      
Image_Path = "Icons\Reference.jpg";                                         % Path of the input image                                              

% Define the parameters for the motion generation execution
EMDM_Path = "EMDM";                                                         % Path name of the EMDM human motion model generation project                                                  
Conda_Env = "emdm";                                                         % Environment name of the EMDM project
Guidance_Param = "5.0";                                                     % Guidance parameter of EMDM. Larger value means more compliance to the input prompt but less creative
Motion_FPS = 20;                                                            % FPS information of the EMDM output

% Define the deterministic random seed to guarantee reproducible physical simulations
Random_Seed = 42;                                                           % Global random seed controlling procedural stochastic terms
rng(Random_Seed, 'twister');

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

% Define the system prompt template to instruct the model for dual task extraction expanding distinct target classes
JSON_Template = sprintf(['{\n', ...
    '  "persons": [\n', ...
    '    {\n', ...
    '      "Refined_Prompt": "...",\n', ...
    '      "start_pos": [0, 0, 0],\n', ...
    '      "start_heading": 0.0,\n', ...
    '      "start_time_delay": 0.0,\n', ...
    '      "height": 1.70,\n', ...
    '      "weight": 70.0\n', ...
    '    }\n', ...
    '  ],\n', ...
    '  "objects": [\n', ...
    '    {\n', ...
    '      "name": "Wooden Desk",\n', ...
    '      "scatter_points": [\n', ...
    '        [2.0, 3.0, 0.75, 0.5],\n', ...
    '        [1.5, 3.0, 0.75, 0.5],\n', ...
    '        [2.5, 3.0, 0.75, 0.5],\n', ...
    '        [2.0, 2.5, 0.75, 0.5],\n', ...
    '        [2.0, 3.5, 0.75, 0.5]\n', ...
    '      ]\n', ...
    '    }\n', ...
    '  ],\n', ...
    '  "fc": 77000000000,\n', ...
    '  "tp": 0.001,\n', ...
    '  "B": 4000000000,\n', ...
    '  "PRF": 1000,\n', ...
    '  "fs": 10000000,\n', ...
    '  "tx_pos": [[0, 5, 1]],\n', ...
    '  "rx_pos": [[0.25, 5, 1], [0.5, 5, 1]],\n', ...
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
System_Prompt = strcat("你现在是一个专业的Human Motion生成与雷达仿真物理参数提取专家。\n"...
    , "你的核心任务是理解用户对'室内人体动作与雷达探测场景'的自然语言描述，并将其精准地转化为结构化的JSON参数字典。\n\n"...
    , "【任务一：多目标动作参数提取 (persons数组)】\n"...
    , "如果场景中包含多个人物，请将他们分离开，并存入 persons 列表数组中（哪怕只有1个人也要放入列表中）。每个 person 需要精准提取以下参数：\n"...
    , "- Refined_Prompt: 将该人物的基础动作描述翻译并润色为HumanML3D描述风格的英文Prompt（10-30词）。对于跌倒、坐下等有明确终态的动作，必须明确指出速度和最终姿态。\n"...
    , "- start_pos: 动作起始三维绝对坐标 [X, Y, Z] (单位：米)。如果没有明确指定，默认值为 [0, 0, 0]。\n"...
    , "- start_heading: 起始运动朝向的偏航角（单位：度，0代表沿X轴正向，90代表沿Y轴正向，默认为0）。\n"...
    , "- start_time_delay: 动作的起始时间间隔或延迟（单位：秒，若未提及默认为0.0）。\n"...
    , "- height: 身高（单位：米，用于计算雷达截面积，若未明确默认1.70）。\n"...
    , "- weight: 体重（单位：公斤，用于计算雷达截面积，若未明确默认70.0）。\n\n"...
    , "【任务二：雷达与物理场景参数提取】\n"...
    , "从描述中提取雷达射频参数、天线位置及场景媒质属性。请严格遵循以下物理单位和坐标系定义（若用户未提供某些参数，请补充合理默认值）：\n"...
    , "- 射频参数（必须转换为基本单位）：中心频率 fc (Hz，例如2GHz必须输出为 2000000000)、带宽 B (Hz，例如1GHz为 1000000000)、脉冲宽度 tp (秒, 默认 1e-3)、脉冲重复频率 PRF (Hz)、采样率 fs (Hz, 默认 10e6)。\n"...
    , "- 系统与天线：系统信噪比 SNR (dB, 默认 50)、天线增益 antenna_gain (dBi, 默认 15)、收发隔离度 antenna_isolation (dB, 默认 40)。\n"...
    , "- 坐标系定义（绝对准则）：所有的三维向量必须严格遵守 [X(方位/宽度), Y(距离/深度/厚度), Z(高度)] 的笛卡尔物理坐标系。\n"...
    , "- 天线位置：发射天线 tx_pos 和接收天线 rx_pos 格式必须为包含 [X, Y, Z] 的二维数组 (单位：米)。哪怕是单天线也必须嵌套一层列表（例如 [[0, 5, 1]]），多发多收则输出如 [[0, 5, 1], [0.25, 5, 1]] 以支持MIMO阵列结构提取。\n"...
    , "- 墙体参数（重要）：若语义中包含墙体则 enable_wall = true，墙体中心 wall_center 为 [X, Y, Z]。**墙体尺寸 wall_dimensions 必须且只能按照 [X轴宽度, Y轴厚度, Z轴高度] 的顺序输出！** 例如用户说'宽5米、高3米、厚0.2米'，必须输出为 [5.0, 0.2, 3.0]！墙体介电常数 wall_epsilon_r 和损耗角正切 wall_loss_tangent 按原意提取。\n"...
    , "- 信号处理：stft_window_seconds (秒, 默认 0.1)，stft_overlap_ratio (0~1之间, 默认 0.75)。\n"...
    , "- 功能开关 (布尔值 true/false)：是否启用墙体 (enable_wall)、是否考虑多径效应 (enable_multipath)、是否使用神经网络进行特征增强/去噪 (enable_network)。如果用户明确说不使用某项功能，设为 false。\n\n"...
    , "【任务三：静止简单物体与关键散射点提取 (objects数组)】\n"...
    , "如果用户描述了房间内另外放置了简单常见物体（如桌子、椅子、沙发、柜子等），请将它们存入 objects 列表数组中（若无此类物体则返回空列表 []）。每个 object 需要提取并推演生成：\n"...
    , "- name: 物体名称的英文描述（如 'Wooden Desk'）。\n"...
    , "- scatter_points: 必须是一个二维数组，每一行代表大模型为你推演出的该物体上的一个关键雷达散射点，格式严格为 [X, Y, Z, RCS]。其中 X/Y/Z 为该点在场景中的绝对三维笛卡尔物理坐标(单位：米)，RCS 为该散射点对应的雷达截面积估值(平方米，典型值0.1-2.0，金属偏大，木质偏小)。\n"...
    , "- 请充分根据描述中物体的物理长宽高尺寸和指定方位，利用你的三维空间几何理解能力，自动为其生成 5 到 15 个分布合理的关键离散散射点（例如一张长1.2米的桌子应包含四个桌角坐标、桌面中心坐标、甚至是桌腿坐标等），从而利用离散点云完美近似勾勒出该物体的三维空间轮廓和电磁反射特征，这对于雷达仿真至关重要。\n\n"...
    , "【输出格式严格约束】\n"...
    , "你的输出必须且只能是一个合法的JSON对象，完全匹配下方模板的键值名称和数据类型。绝对不要输出任何额外的思考过程、问候语或 ```json 等Markdown标记！\n"...
    , "模板示例：\n", JSON_Template); % The system prompt consists of distinct parts naturally appending capabilities without erasing structural foundations natively

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
    
    % Automatically clean mathematical expressions and format flaws safely preserving arrays structures
    json_str = clean_json_math(char(json_str));
    
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
    
    % Apply robust multi-antenna spatial extraction correctly parsing formats safely avoiding LLM hallucination
    Simulation_Params.tx_pos = parse_antenna_positions(Simulation_Params.tx_pos);
    Simulation_Params.rx_pos = parse_antenna_positions(Simulation_Params.rx_pos);

    % Ensure persons array exists and handles struct array conversions safely
    if isfield(Simulation_Params, 'persons')
        if iscell(Simulation_Params.persons)
            % Homogenize fields across all cells avoiding 'struct contents differ' concatenation errors
            all_fields = {};
            for i_p = 1:length(Simulation_Params.persons)
                all_fields = union(all_fields, fieldnames(Simulation_Params.persons{i_p}));
            end
            struct_array = struct();
            for i_p = 1:length(Simulation_Params.persons)
                for f = 1:length(all_fields)
                    if isfield(Simulation_Params.persons{i_p}, all_fields{f})
                        struct_array(i_p).(all_fields{f}) = Simulation_Params.persons{i_p}.(all_fields{f});
                    else
                        struct_array(i_p).(all_fields{f}) = [];
                    end
                end
            end
            Simulation_Params.persons = struct_array;
        end
    else
        if isfield(Simulation_Params, 'Refined_Prompt')
            % LLM ignored persons array request, wrap it securely as a single target
            Simulation_Params.persons = Simulation_Params;
        else
            error("The language model did not return a valid persons multi-target array.");
        end
    end
    
    % Fill missing properties inside multi-person extraction structure robustly
    num_persons = length(Simulation_Params.persons);
    for i_p = 1:num_persons
        if ~isfield(Simulation_Params.persons(i_p), 'start_pos') || isempty(Simulation_Params.persons(i_p).start_pos)
            Simulation_Params.persons(i_p).start_pos = [0, 0, 0];
        end
        if ~isfield(Simulation_Params.persons(i_p), 'start_heading') || isempty(Simulation_Params.persons(i_p).start_heading)
            Simulation_Params.persons(i_p).start_heading = 0.0;
        end
        if ~isfield(Simulation_Params.persons(i_p), 'start_time_delay') || isempty(Simulation_Params.persons(i_p).start_time_delay)
            Simulation_Params.persons(i_p).start_time_delay = 0.0;
        end
        if ~isfield(Simulation_Params.persons(i_p), 'height') || isempty(Simulation_Params.persons(i_p).height)
            Simulation_Params.persons(i_p).height = 1.70;
        end
        if ~isfield(Simulation_Params.persons(i_p), 'weight') || isempty(Simulation_Params.persons(i_p).weight)
            Simulation_Params.persons(i_p).weight = 70.0;
        end
    end

    % Extract environment objects matrix handling struct array definitions resolving LLM complex nesting safely securely tracking layouts
    if ~isfield(Simulation_Params, 'objects')
        Simulation_Params.objects = [];
    end
    if ~isempty(Simulation_Params.objects)
        if iscell(Simulation_Params.objects)
            all_obj_fields = {};
            for i_o = 1:length(Simulation_Params.objects)
                all_obj_fields = union(all_obj_fields, fieldnames(Simulation_Params.objects{i_o}));
            end
            obj_array = struct();
            for i_o = 1:length(Simulation_Params.objects)
                for f = 1:length(all_obj_fields)
                    if isfield(Simulation_Params.objects{i_o}, all_obj_fields{f})
                        obj_array(i_o).(all_obj_fields{f}) = Simulation_Params.objects{i_o}.(all_obj_fields{f});
                    else
                        obj_array(i_o).(all_obj_fields{f}) = [];
                    end
                end
            end
            Simulation_Params.objects = obj_array;
        end
        % Sanitize scattering points array representations formatting limits naturally preventing crashes natively correctly
        for i_o = 1:length(Simulation_Params.objects)
            if isfield(Simulation_Params.objects(i_o), 'scatter_points') && ~isempty(Simulation_Params.objects(i_o).scatter_points)
                sp_data = Simulation_Params.objects(i_o).scatter_points;
                if iscell(sp_data)
                    sp_mat = [];
                    for r_idx = 1:length(sp_data)
                        if iscell(sp_data{r_idx})
                            sp_mat = [sp_mat; cell2mat(sp_data{r_idx})];
                        elseif isnumeric(sp_data{r_idx})
                            sp_mat = [sp_mat; sp_data{r_idx}(:)'];
                        end
                    end
                    if size(sp_mat, 2) == 3
                        sp_mat = [sp_mat, repmat(0.5, size(sp_mat, 1), 1)];
                    end
                    Simulation_Params.objects(i_o).scatter_points = double(sp_mat);
                elseif isnumeric(sp_data)
                    if size(sp_data, 2) == 3
                        sp_data = [sp_data, repmat(0.5, size(sp_data, 1), 1)];
                    end
                    Simulation_Params.objects(i_o).scatter_points = double(sp_data);
                else
                    Simulation_Params.objects(i_o).scatter_points = [];
                end
            else
                Simulation_Params.objects(i_o).scatter_points = [];
            end
            if ~isfield(Simulation_Params.objects(i_o), 'name') || isempty(Simulation_Params.objects(i_o).name)
                Simulation_Params.objects(i_o).name = 'Unknown Object';
            end
        end
    end

    % Apply deeper sanitation to all extracted physical parameters guaranteeing numeric and geometric consistency
    Simulation_Params = sanitize_simulation_params(Simulation_Params, Default_Params);
    num_persons = length(Simulation_Params.persons);

    % Replace default print with agent thinking stream
    agent_think(sprintf("Successfully extracted MIMO hardware properties, refined %d semantic motion prompts and constructed %d environmental object geometries natively.", num_persons, length(Simulation_Params.objects)));
    
    %% Call EMDM to Generate 3D Motion Data
    % Save the original directory path and switch to the motion generation directory
    original_dir = pwd;
    if ~isfolder(EMDM_Path)
        error("The specified EMDM motion generation directory does not exist.");
    end
    cd(EMDM_Path);
    dir_guard = onCleanup(@() cd(original_dir));
    
    % Iterate over all defined persons generating distinct motions dynamically
    for i_p = 1:num_persons
        agent_think(sprintf("Delegating the three dimensional human motion trajectory generation for Target %d to the deep diffusion mathematical framework.", i_p));

        % Clean the generated prompt text for command line usage
        if ~isfield(Simulation_Params.persons(i_p), 'Refined_Prompt') || strlength(strtrim(string(Simulation_Params.persons(i_p).Refined_Prompt))) == 0
            Simulation_Params.persons(i_p).Refined_Prompt = "a person stands naturally and remains still";
        end
        Cleaned_Prompt = string(Simulation_Params.persons(i_p).Refined_Prompt);
        Cleaned_Prompt = regexprep(Cleaned_Prompt, '[\r\n]+', ' ');
        Cleaned_Prompt = strrep(Cleaned_Prompt, '"', '');
        Cleaned_Prompt = strrep(Cleaned_Prompt, '''', '');

        % Scan the current directory to track existing results files locally
        pre_run_files = dir(fullfile('models', '**', 'results.npy'));
        pre_run_paths = string(fullfile({pre_run_files.folder}, {pre_run_files.name}));
        run_start_time = now;

        % Construct the execution command for the conda environment
        model_pth = fullfile('models', 'HumanML3D.pth');
        cmd = sprintf('conda run -n %s python sample_mdm.py --text_prompt "%s" --model_path "%s" --dataset humanml --guidance_param %s', ...
                      Conda_Env, Cleaned_Prompt, model_pth, Guidance_Param);

        % Replace default print with agent thinking stream
        agent_think(sprintf("Executing system shell command for Target %d. Triggering the external python inference kernel... Command: %s", i_p, cmd));

        % Run the motion generation command through the system shell while tolerating non critical external warnings
        [status_gen, cmdout_gen] = system(cmd);

        %% Find & Load Generated Results
        % Scan the directory again to locate the newly generated or updated results
        latest_npy = select_latest_results_file(pre_run_paths, fullfile('models', '**', 'results.npy'), run_start_time);

        % Throw an error only if no usable motion data file is found
        if strlength(latest_npy) == 0
            if status_gen ~= 0
                fprintf('%s\n', cmdout_gen);
            end
            agent_think(sprintf("Action generation failed for Target %d. No valid resulting matrix was detected within the system pipeline.", i_p));
            error("Action generation failed. No new or updated resulting array was generated. Skipping visualization.");
        elseif status_gen ~= 0
            % External python environments may emit non critical warnings or non zero statuses while still generating valid motion files
            agent_think(sprintf("The external motion generation environment returned a non zero status for Target %d, but a valid resulting array was still detected and will be used.", i_p));
        end

        % Define temporary script names for the python conversion tool dynamically tracking instances
        temp_py = sprintf('temp_npy2mat_%d.py', i_p);
        temp_mat = sprintf('temp_motion_%d.mat', i_p);

        % Write the python script to safely convert the numpy array into a matlab readable format
        fid_py = fopen(temp_py, 'w');
        if fid_py < 0
            error("Unable to create the temporary python conversion script for motion matrix translation.");
        end
        fprintf(fid_py, 'import sys\nimport numpy as np\nimport scipy.io as sio\n');
        fprintf(fid_py, 'npy_path = sys.argv[1]\nmat_path = sys.argv[2]\n');
        fprintf(fid_py, 'data = np.load(npy_path, allow_pickle=True)\n');
        fprintf(fid_py, 'if isinstance(data, np.ndarray) and data.shape == (): data = data.item()\n');
        fprintf(fid_py, 'motion = data.get("motion", data) if isinstance(data, dict) else data\n');
        fprintf(fid_py, 'motion = np.asarray(motion)\n');
        fprintf(fid_py, 'motion = np.squeeze(motion)\n');
        fprintf(fid_py, 'while motion.ndim > 3: motion = motion[0]\n');
        fprintf(fid_py, 'if motion.ndim == 3 and motion.shape[0] == 22 and motion.shape[1] == 3: motion = np.transpose(motion, (2, 0, 1))\n');
        fprintf(fid_py, 'elif motion.ndim == 3 and motion.shape[0] == 3 and motion.shape[1] == 22: motion = np.transpose(motion, (2, 1, 0))\n');
        fprintf(fid_py, 'elif motion.ndim == 3 and motion.shape[1] == 3 and motion.shape[2] == 22: motion = np.transpose(motion, (0, 2, 1))\n');
        fprintf(fid_py, 'sio.savemat(mat_path, {"motion": motion})\n');
        fclose(fid_py);

        % Execute the python conversion script inside the conda environment
        latest_npy_char = char(latest_npy);
        conv_cmd = sprintf('conda run -n %s python "%s" "%s" "%s"', Conda_Env, temp_py, latest_npy_char, temp_mat);
        [status_conv, cmdout_conv] = system(conv_cmd);

        % Embed a rigorous file generation verification preventing silent execution failures crashing subsequent loadings
        if ~isfile(temp_mat)
            if isfile(temp_py), delete(temp_py); end
            if status_conv ~= 0
                fprintf('%s\n', cmdout_conv);
            end
            agent_think("Data format conversion pipeline collapsed. Python process failed to bridge external structures.");
            error("Python conversion failed to create the target temporary data file. Please investigate python and deep learning environments.");
        elseif status_conv ~= 0
            agent_think(sprintf("The temporary numpy to matlab conversion routine for Target %d emitted a non zero status, but the converted matrix was generated successfully and will be used.", i_p));
        end

        % Load the converted motion matrix data into the workspace
        loaded_data = load(temp_mat);
        if ~isfield(loaded_data, 'motion')
            delete(temp_py);
            delete(temp_mat);
            error("The converted temporary data file does not contain a readable motion matrix.");
        end
        motion_mat = sanitize_motion_matrix(double(loaded_data.motion));

        % Delete the temporary files cleanly releasing memory
        delete(temp_py);
        delete(temp_mat);

        %% Post-Processing to Improve Target Motion Quality
        % Replace default print with agent thinking stream
        agent_think(sprintf("Applying kinematic smoothing, orientation mappings, spatial translations and temporal bounds correctly to Target %d.", i_p));

        % Apply filter parameters to smooth the chaotic motion trajectories
        window_size = robust_sg_window(size(motion_mat, 1), 9);
        poly_order = min(3, max(window_size - 2, 1));
        if window_size >= 5
            for j = 1:size(motion_mat, 2)
                for d = 1:size(motion_mat, 3)
                    if all(isfinite(motion_mat(:, j, d)))
                        motion_mat(:, j, d) = sgolayfilt(motion_mat(:, j, d), poly_order, window_size);
                    end
                end
            end
        end

        % Calculate relative velocity to find the active frames
        orig_frames = size(motion_mat, 1);
        root_pos = motion_mat(:, 1, :);
        rel_motion = motion_mat - root_pos;
        rel_vel = diff(rel_motion, 1, 1); 

        % Calculate the overall frame activity energy to detect the idle state
        frame_activity = squeeze(sum(sum(rel_vel.^2, 2), 3));
        if isempty(frame_activity) || ~any(isfinite(frame_activity))
            frame_activity = zeros(max(orig_frames - 1, 1), 1);
        end
        idle_threshold = max(1e-5, 0.02 * max(frame_activity)); 
        is_active = frame_activity > idle_threshold;

        % Find the last frame where the human skeleton remains active
        last_active_frame = orig_frames;
        for f = orig_frames - 1 : -1 : 1
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

        % Center the motion at the origin (X and Depth only) based on the first frame's root joint natively mapping offsets
        root_x0 = motion_mat(1, 1, 1);
        root_y0 = motion_mat(1, 1, 3); 

        yaw_rad = deg2rad(mod(Simulation_Params.persons(i_p).start_heading, 360));
        start_pos = ensure_row_vector(Simulation_Params.persons(i_p).start_pos, [0, 0, 0], 3);

        % EMDM natively moves forward along its Y axis (Dim 3).
        % We map heading=0 to Radar +X, heading=90 to Radar +Y.
        % This corresponds to a 2D rotation of (heading - 90) degrees smoothly translating vectors mathematically.
        rot_ang = yaw_rad - pi/2;
        cos_r = cos(rot_ang);
        sin_r = sin(rot_ang);

        % Vectorize spatial rotation and translation to reduce numerical and computational overhead
        x_centered = motion_mat(:, :, 1) - root_x0;
        y_centered = motion_mat(:, :, 3) - root_y0;
        motion_mat(:, :, 1) = x_centered * cos_r - y_centered * sin_r + start_pos(1);
        motion_mat(:, :, 3) = x_centered * sin_r + y_centered * cos_r + start_pos(2);
        motion_mat(:, :, 2) = motion_mat(:, :, 2) + start_pos(3);

        % Integrate starting temporal delay securely holding the starting stance correctly shifting timelines
        delay_sec = max(0.0, double(Simulation_Params.persons(i_p).start_time_delay));
        delay_frames = round(delay_sec * Motion_FPS);
        if delay_frames > 0
            pad_front = repmat(motion_mat(1, :, :), [delay_frames, 1, 1]);
            motion_mat = cat(1, pad_front, motion_mat);
        end

        % Record a person specific horizontal body radius improving downstream collision realism
        Simulation_Params.persons(i_p).collision_radius = estimate_person_footprint_radius(motion_mat);

        % Embed modified kinematic data structurally safely returning isolated sequence back
        Simulation_Params.persons(i_p).motion_mat = motion_mat;
    end

    % Restore original operational environment
    clear dir_guard;
    cd(original_dir); 
    
    %% Consolidate & Pad Global Target Timelines
    % Replace default print with agent thinking stream
    agent_think("Aligning disparate temporal length trajectories across all detected physical entities appending steady states.");
    
    max_frames = 0;
    for i_p = 1:num_persons
        if size(Simulation_Params.persons(i_p).motion_mat, 1) > max_frames
            max_frames = size(Simulation_Params.persons(i_p).motion_mat, 1);
        end
    end
    
    for i_p = 1:num_persons
        curr_frames = size(Simulation_Params.persons(i_p).motion_mat, 1);
        if curr_frames < max_frames
            pad_len = max_frames - curr_frames;
            pad_mat = repmat(Simulation_Params.persons(i_p).motion_mat(end, :, :), [pad_len, 1, 1]);
            Simulation_Params.persons(i_p).motion_mat = cat(1, Simulation_Params.persons(i_p).motion_mat, pad_mat);
        end
    end
    
    %% Post-Processing: Spatial-Temporal Collision Avoidance and Constraint Enforcement
    % Replace default print with agent thinking stream
    agent_think("Evaluating unified temporal physical bounds to calculate dynamic collision models preventing intersecting human kinematics and geometric environment penetrations.");
    
    % Define physical safety boundary parameters preventing overlapping instances smoothly
    wall_margin = 0.1;                                                      % Repulsive boundary buffering wall geometries safely

    % Resolve static wall geometric boundaries rigorously mapping limits
    wall_bbox = [-inf, inf, -inf, inf];
    if Simulation_Params.enable_wall
        w_c = Simulation_Params.wall_center;
        w_d = Simulation_Params.wall_dimensions;

        % Determine the extended two dimensional structural footprint expanding physical limits safely
        wall_bbox = [w_c(1) - w_d(1)/2, w_c(1) + w_d(1)/2, w_c(2) - w_d(2)/2, w_c(2) + w_d(2)/2];
    end

    % Precompute projected static object footprints accelerating the frame wise collision solver
    object_bboxes = zeros(0, 4);
    if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
        for i_o = 1:length(Simulation_Params.objects)
            sp_matrix = Simulation_Params.objects(i_o).scatter_points;
            if ~isempty(sp_matrix) && size(sp_matrix, 2) >= 3
                object_bboxes(end + 1, :) = [min(sp_matrix(:, 1)), max(sp_matrix(:, 1)), min(sp_matrix(:, 2)), max(sp_matrix(:, 2))];
            end
        end
    end

    % Extract person specific collision radii from the kinematic envelopes improving physical realism
    person_radius = zeros(1, num_persons);
    for i_p = 1:num_persons
        if isfield(Simulation_Params.persons(i_p), 'collision_radius') && isfinite(Simulation_Params.persons(i_p).collision_radius)
            person_radius(i_p) = Simulation_Params.persons(i_p).collision_radius;
        else
            person_radius(i_p) = estimate_person_footprint_radius(Simulation_Params.persons(i_p).motion_mat);
        end
    end

    % Pre-solve initial frame static overlaps pushing conflicting entities completely apart globally instantly
    initial_off_X = zeros(1, num_persons);
    initial_off_Y = zeros(1, num_persons);
    for iter = 1:15
        curr_X = zeros(1, num_persons);
        curr_Y = zeros(1, num_persons);
        for i_p = 1:num_persons
            curr_X(i_p) = Simulation_Params.persons(i_p).motion_mat(1, 1, 1) + initial_off_X(i_p);
            curr_Y(i_p) = Simulation_Params.persons(i_p).motion_mat(1, 1, 3) + initial_off_Y(i_p);
        end
        
        for i_p = 1:num_persons
            if Simulation_Params.enable_wall
                [dx, dy, is_pen] = project_circle_out_of_bbox(curr_X(i_p), curr_Y(i_p), person_radius(i_p) + wall_margin, wall_bbox);
                if is_pen
                    initial_off_X(i_p) = initial_off_X(i_p) + dx;
                    initial_off_Y(i_p) = initial_off_Y(i_p) + dy;
                    curr_X(i_p) = curr_X(i_p) + dx;
                    curr_Y(i_p) = curr_Y(i_p) + dy;
                end
            end
            if ~isempty(object_bboxes)
                for i_o = 1:size(object_bboxes, 1)
                    [dx, dy, is_pen] = project_circle_out_of_bbox(curr_X(i_p), curr_Y(i_p), person_radius(i_p) + wall_margin, object_bboxes(i_o, :));
                    if is_pen
                        initial_off_X(i_p) = initial_off_X(i_p) + dx;
                        initial_off_Y(i_p) = initial_off_Y(i_p) + dy;
                        curr_X(i_p) = curr_X(i_p) + dx;
                        curr_Y(i_p) = curr_Y(i_p) + dy;
                    end
                end
            end
        end
        % Inter-person
        for i_p = 1:num_persons
            for j_p = i_p+1:num_persons
                vec = [curr_X(i_p) - curr_X(j_p), curr_Y(i_p) - curr_Y(j_p)];
                dist = norm(vec);
                min_dist = person_radius(i_p) + person_radius(j_p);
                if dist < min_dist
                    if dist <= 1e-4
                        seed_angle = 2 * pi * (i_p + j_p + iter) / 100;
                        push = [cos(seed_angle), sin(seed_angle)] * (min_dist / 2.0);
                    else
                        overlap = min_dist - dist;
                        push = (vec / dist) * (overlap / 2.0);
                    end
                    initial_off_X(i_p) = initial_off_X(i_p) + push(1);
                    initial_off_Y(i_p) = initial_off_Y(i_p) + push(2);
                    initial_off_X(j_p) = initial_off_X(j_p) - push(1);
                    initial_off_Y(j_p) = initial_off_Y(j_p) - push(2);
                    curr_X(i_p) = curr_X(i_p) + push(1);
                    curr_Y(i_p) = curr_Y(i_p) + push(2);
                    curr_X(j_p) = curr_X(j_p) - push(1);
                    curr_Y(j_p) = curr_Y(j_p) - push(2);
                end
            end
        end
    end
    
    % Apply base offsets globally mapping all timeline configurations instantly preventing sliding artifacts entirely
    for i_p = 1:num_persons
        if initial_off_X(i_p) ~= 0 || initial_off_Y(i_p) ~= 0
            Simulation_Params.persons(i_p).motion_mat(:, :, 1) = Simulation_Params.persons(i_p).motion_mat(:, :, 1) + initial_off_X(i_p);
            Simulation_Params.persons(i_p).motion_mat(:, :, 3) = Simulation_Params.persons(i_p).motion_mat(:, :, 3) + initial_off_Y(i_p);
        end
    end

    % Allocate continuous spatial offset tracking arrays securely buffering translations
    offset_X = zeros(max_frames, num_persons);
    offset_Y = zeros(max_frames, num_persons);

    % Execute iterative physics solver predicting, bounding and mapping trajectories chronologically
    for f = 1:max_frames
        % Step 1: Predict current frame absolute positions based on previous frame actuals preventing drifts
        if f > 1
            offset_X(f, :) = offset_X(f - 1, :);
            offset_Y(f, :) = offset_Y(f - 1, :);
        end

        % Step 2: Iterative relaxation for spatial conflicts ensuring rigid constraint convergence
        num_relaxations = 6;
        for iter = 1:num_relaxations
            % Extract the evaluated physical coordinate positions dynamically at the current iteration
            curr_roots_X = zeros(num_persons, 1);
            curr_roots_Y = zeros(num_persons, 1);
            for i_p = 1:num_persons
                curr_roots_X(i_p) = Simulation_Params.persons(i_p).motion_mat(f, 1, 1) + offset_X(f, i_p);
                curr_roots_Y(i_p) = Simulation_Params.persons(i_p).motion_mat(f, 1, 3) + offset_Y(f, i_p);
            end

            % Component A: Structural Wall Collisions Deflection
            if Simulation_Params.enable_wall
                for i_p = 1:num_persons
                    [delta_x_wall, delta_y_wall, is_wall_penetrating] = project_circle_out_of_bbox(curr_roots_X(i_p), curr_roots_Y(i_p), person_radius(i_p) + wall_margin, wall_bbox);
                    if is_wall_penetrating
                        offset_X(f, i_p) = offset_X(f, i_p) + delta_x_wall;
                        offset_Y(f, i_p) = offset_Y(f, i_p) + delta_y_wall;
                        curr_roots_X(i_p) = curr_roots_X(i_p) + delta_x_wall;
                        curr_roots_Y(i_p) = curr_roots_Y(i_p) + delta_y_wall;
                    end
                end
            end

            % Component B: Static Objects Collisions Deflection evaluating bounds preventing phantom overlaps
            if ~isempty(object_bboxes)
                for i_o = 1:size(object_bboxes, 1)
                    for i_p = 1:num_persons
                        [delta_x_obj, delta_y_obj, is_obj_penetrating] = project_circle_out_of_bbox(curr_roots_X(i_p), curr_roots_Y(i_p), person_radius(i_p) + wall_margin, object_bboxes(i_o, :));
                        if is_obj_penetrating
                            offset_X(f, i_p) = offset_X(f, i_p) + delta_x_obj;
                            offset_Y(f, i_p) = offset_Y(f, i_p) + delta_y_obj;
                            curr_roots_X(i_p) = curr_roots_X(i_p) + delta_x_obj;
                            curr_roots_Y(i_p) = curr_roots_Y(i_p) + delta_y_obj;
                        end
                    end
                end
            end

            % Component C: Dynamic Inter-person Collisions Repulsion
            for i_p = 1:num_persons
                for j_p = i_p + 1:num_persons
                    xi = curr_roots_X(i_p);
                    yi = curr_roots_Y(i_p);
                    xj = curr_roots_X(j_p);
                    yj = curr_roots_Y(j_p);

                    vec = [xi - xj, yi - yj];
                    dist = norm(vec);
                    person_min_dist = person_radius(i_p) + person_radius(j_p);

                    if dist < person_min_dist && dist > 1e-4
                        overlap = person_min_dist - dist;
                        push = (vec / dist) * (overlap / 2.0);

                        offset_X(f, i_p) = offset_X(f, i_p) + push(1);
                        offset_Y(f, i_p) = offset_Y(f, i_p) + push(2);
                        offset_X(f, j_p) = offset_X(f, j_p) - push(1);
                        offset_Y(f, j_p) = offset_Y(f, j_p) - push(2);
                    elseif dist <= 1e-4
                        % Disambiguate exact overlapping coordinates utilizing a deterministic micro phase securely
                        seed_angle = 2 * pi * (i_p + j_p + f) / max(num_persons + max_frames, 1);
                        push = [cos(seed_angle), sin(seed_angle)] * (person_min_dist / 2.0);

                        offset_X(f, i_p) = offset_X(f, i_p) + push(1);
                        offset_Y(f, i_p) = offset_Y(f, i_p) + push(2);
                        offset_X(f, j_p) = offset_X(f, j_p) - push(1);
                        offset_Y(f, j_p) = offset_Y(f, j_p) - push(2);
                    end

                    % Sync local variables immediately avoiding using stale coordinates causing phantom drifts
                    curr_roots_X(i_p) = Simulation_Params.persons(i_p).motion_mat(f, 1, 1) + offset_X(f, i_p);
                    curr_roots_Y(i_p) = Simulation_Params.persons(i_p).motion_mat(f, 1, 3) + offset_Y(f, i_p);
                    curr_roots_X(j_p) = Simulation_Params.persons(j_p).motion_mat(f, 1, 1) + offset_X(f, j_p);
                    curr_roots_Y(j_p) = Simulation_Params.persons(j_p).motion_mat(f, 1, 3) + offset_Y(f, j_p);
                end
            end
        end
    end

    % Step 3: Apply temporal smoothing filtering bounding sudden positional leaps eliminating Doppler artifacts completely
    agent_think("Applying temporal Savitzky-Golay filters universally harmonizing collision avoidance shifts to completely prevent unnatural micro-Doppler kinematic anomalies.");
    smooth_window = min(31, max_frames - mod(max_frames, 2) - 1);
    if smooth_window >= 5
        for i_p = 1:num_persons
            offset_X(:, i_p) = sgolayfilt(offset_X(:, i_p), 3, smooth_window);
            offset_Y(:, i_p) = sgolayfilt(offset_Y(:, i_p), 3, smooth_window);
        end
    end
    
    % Step 4: Reproject the smoothed offsets once more guaranteeing that smoothing does not reintroduce penetrations
    for f = 1:max_frames
        for i_p = 1:num_persons
            curr_x = Simulation_Params.persons(i_p).motion_mat(f, 1, 1) + offset_X(f, i_p);
            curr_y = Simulation_Params.persons(i_p).motion_mat(f, 1, 3) + offset_Y(f, i_p);

            if Simulation_Params.enable_wall
                [delta_x_wall, delta_y_wall, is_wall_penetrating] = project_circle_out_of_bbox(curr_x, curr_y, person_radius(i_p) + wall_margin, wall_bbox);
                if is_wall_penetrating
                    offset_X(f, i_p) = offset_X(f, i_p) + delta_x_wall;
                    offset_Y(f, i_p) = offset_Y(f, i_p) + delta_y_wall;
                    curr_x = curr_x + delta_x_wall;
                    curr_y = curr_y + delta_y_wall;
                end
            end

            if ~isempty(object_bboxes)
                for i_o = 1:size(object_bboxes, 1)
                    [delta_x_obj, delta_y_obj, is_obj_penetrating] = project_circle_out_of_bbox(curr_x, curr_y, person_radius(i_p) + wall_margin, object_bboxes(i_o, :));
                    if is_obj_penetrating
                        offset_X(f, i_p) = offset_X(f, i_p) + delta_x_obj;
                        offset_Y(f, i_p) = offset_Y(f, i_p) + delta_y_obj;
                        curr_x = curr_x + delta_x_obj;
                        curr_y = curr_y + delta_y_obj;
                    end
                end
            end
        end
    end

    % Step 5: Apply the bounded translations strictly aligning skeletal configurations mapping equally across all joint arrays natively
    for i_p = 1:num_persons
        % Utilize implicit expansion reshape dimensions securely adding offsets natively mapping vectors across all 22 components
        off_x = reshape(offset_X(:, i_p), [max_frames, 1, 1]);
        off_y = reshape(offset_Y(:, i_p), [max_frames, 1, 1]);
        
        Simulation_Params.persons(i_p).motion_mat(:, :, 1) = Simulation_Params.persons(i_p).motion_mat(:, :, 1) + off_x;
        Simulation_Params.persons(i_p).motion_mat(:, :, 3) = Simulation_Params.persons(i_p).motion_mat(:, :, 3) + off_y;
    end

    %% Build and Save the Comprehensive Simulation_Params Struct
    % Replace default print with agent thinking stream
    agent_think("Consolidating the extracted hardware properties, target kinematics and RCS properties into the persistent physical parameter collection system.");
    
    Simulation_Params.num_frames = max_frames;            
    Simulation_Params.num_joints = 22;            
    
    % Append temporal information crucial for the doppler shift calculation universally
    Simulation_Params.FPS = Motion_FPS;                            
    Simulation_Params.time_axis = (0 : Simulation_Params.num_frames - 1) / Motion_FPS; 
    
    % Define the standard kinematic tree topology for bone connecting
    kinematic_tree =[
        1, 2; 1, 3; 1, 4; 2, 5; 3, 6; 4, 7; 5, 8; 6, 9; 
        7, 10; 8, 11; 9, 12; 10, 13; 10, 14; 10, 15; 13, 16; 
        14, 17; 15, 18; 17, 19; 18, 20; 19, 21; 20, 22
    ];
    Simulation_Params.kinematic_tree = kinematic_tree;
    
    % Define the normalized radar cross section values for each body joint independently
    Normalized_RCS =[1.0, 0.7, 0.7, 0.9, 0.5, 0.5, 0.8, 0.3, 0.3, 0.8, ...
                      0.1, 0.1, 0.6, 0.5, 0.5, 0.85, 0.4, 0.4, 0.3, 0.3, ...
                      0.1, 0.1];
    Simulation_Params.Normalized_RCS = Normalized_RCS;
    
    % Map the normalized values to spatial physical radii for the drawing templates consistently
    min_joint_radius = 0.02;
    max_joint_radius = 0.075;
    joint_radii = min_joint_radius + (max_joint_radius - min_joint_radius) * Normalized_RCS;
    
    % Store the calculated baseline radii settings into the main structure
    Simulation_Params.joint_radii = joint_radii;                   
    Simulation_Params.bone_radius = 0.035;                         
    
    % Compute physical proportions allocating strict RCS scaling scaling mathematical boundaries reliably
    base_hw_factor = sqrt(1.70 * 70.0);
    for i_p = 1:num_persons
        H = Simulation_Params.persons(i_p).height;
        W = Simulation_Params.persons(i_p).weight;
        % Calculate specific proportional RCS shift rigorously scaling physical signatures smoothly
        RCS_scale = sqrt(H * W) / base_hw_factor;
        
        Simulation_Params.persons(i_p).RCS_scale = RCS_scale;
        Simulation_Params.persons(i_p).Scaled_RCS = Simulation_Params.Normalized_RCS * RCS_scale;
        
        % Visually map radii representations structurally tracking volumetric proportions accurately tracking geometries
        radius_scale = sqrt(RCS_scale);
        Simulation_Params.persons(i_p).joint_radii = Simulation_Params.joint_radii * radius_scale;
        Simulation_Params.persons(i_p).bone_radius = Simulation_Params.bone_radius * radius_scale;
    end
    
    % Save the fully packed simulation parameters struct into the current directory
    save(fullfile(Save_Dir, 'Simulation_Params.mat'), 'Simulation_Params');

    %% FMCW Radar Point-Scatterer Simulation (MIMO Supported)
    % Replace default print with agent thinking stream
    agent_think("Engaging the frequency modulated continuous wave radar physics engine to computationally simulate superimposed point scatterers and complex wall interactions across multiple targets and MIMO channels simultaneously.");
    
    % Extract the radar parameters from the simulation settings
    c = 3e8;                                            
    fc = Simulation_Params.fc;
    lambda = c / fc; % Parse the radar wavelength for proper decay calculations
    tp = Simulation_Params.tp;
    B = Simulation_Params.B;
    K = B / tp;                                         
    PRF = Simulation_Params.PRF;
    fs = Simulation_Params.fs;
    
    % Incorporate antenna gain calculating absolute physics scaled amplitudes
    if isfield(Simulation_Params, 'antenna_gain')
        G_lin = 10^(Simulation_Params.antenna_gain / 10);
    else
        G_lin = 10^(10 / 10); % Fallback default generic indoor 10dBi patch array
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
    eps_r = 1.0;
    
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
    
    % Extract the spatial coordinates of the transmitting and receiving antennas natively resolving MIMO lists
    tx_pos_list = Simulation_Params.tx_pos;              
    rx_pos_list = Simulation_Params.rx_pos;
    num_tx = size(tx_pos_list, 1);
    num_rx = size(rx_pos_list, 1);
    num_channels = num_tx * num_rx;
    
    % Configure the fast time samples and interval duration
    N_s = round(tp * fs);                               
    t_fast = (0:N_s-1)' / fs;
    
    % Calculate the slow time axis based on the pulse repetition frequency
    t_motion = Simulation_Params.time_axis;             
    t_slow = 0 : 1/PRF : t_motion(end);                 
    num_pulses = length(t_slow);
    
    % Interpolate the joint trajectories over the slow time axis parsing independent objects into 4D struct cleanly mapped avoiding arrays clashes
    motion_radar_all = zeros(num_pulses, num_persons, 22, 3);
    for i_p = 1:num_persons
        motion_mat = Simulation_Params.persons(i_p).motion_mat;
        for j = 1:22
            % Use piecewise cubic hermite interpolating polynomial (pchip) natively preserving monotonicity entirely bounded preventing overlaps
            motion_radar_all(:, i_p, j, 1) = interp1(t_motion, motion_mat(:,j,1), t_slow, 'pchip', 'extrap'); 
            motion_radar_all(:, i_p, j, 2) = interp1(t_motion, motion_mat(:,j,2), t_slow, 'pchip', 'extrap'); 
            motion_radar_all(:, i_p, j, 3) = interp1(t_motion, motion_mat(:,j,3), t_slow, 'pchip', 'extrap'); 
        end
    end
    
    % Compute the boresight direction vectors dynamically pointing roughly towards the center of the room for beam evaluation natively accommodating array lists
    scene_center = estimate_scene_center(Simulation_Params, tx_pos_list, rx_pos_list);
    tx_boresight_list = zeros(num_tx, 3);
    for i_tx = 1:num_tx
        tb = scene_center - tx_pos_list(i_tx, :);
        if norm(tb) > 1e-3, tx_boresight_list(i_tx, :) = tb / norm(tb); else, tx_boresight_list(i_tx, :) = [0, -1, 0]; end
    end
    rx_boresight_list = zeros(num_rx, 3);
    for i_rx = 1:num_rx
        rb = scene_center - rx_pos_list(i_rx, :);
        if norm(rb) > 1e-3, rx_boresight_list(i_rx, :) = rb / norm(rb); else, rx_boresight_list(i_rx, :) = [0, -1, 0]; end
    end
    
    % Precompute static environment object echoes rigorously decoupling identical temporal traces significantly compressing execution limits safely
    static_echo_matrix = complex(zeros(N_s, num_channels));
    if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
        agent_think("Precomputing highly dense static background electromagnetic scattering clusters preserving exact coherent phase arrays efficiently...");
        
        for ch = 1:num_channels
            tx_idx = ceil(ch / num_rx);
            rx_idx = mod(ch - 1, num_rx) + 1;
            tx_pos_curr = tx_pos_list(tx_idx, :);
            rx_pos_curr = rx_pos_list(rx_idx, :);
            tx_boresight_curr = tx_boresight_list(tx_idx, :);
            rx_boresight_curr = rx_boresight_list(rx_idx, :);
            
            signal_static_ch = complex(zeros(N_s, 1));
            
            for i_o = 1:length(Simulation_Params.objects)
                sp_matrix = Simulation_Params.objects(i_o).scatter_points;
                if isempty(sp_matrix) || size(sp_matrix, 2) < 4, continue; end
                
                num_sp = size(sp_matrix, 1);
                
                % Seed random independently per object ensuring identical phases align exactly over MIMO channels maintaining spatial coherence naturally
                rng(1000 + i_o);
                obj_random_phases = 2 * pi * rand(num_sp, 1);
                
                for p_idx = 1:num_sp
                    target_pos = sp_matrix(p_idx, 1:3);
                    current_rcs_val = sp_matrix(p_idx, 4);
                    
                    % Calculate distances
                    R_tx = norm(target_pos - tx_pos_curr);
                    R_rx = norm(target_pos - rx_pos_curr);
                    R_total = R_tx + R_rx;
                    
                    % Patterns
                    v_tx = (target_pos - tx_pos_curr) / (R_tx + 1e-6);
                    v_rx = (target_pos - rx_pos_curr) / (R_rx + 1e-6);
                    pat_tx = max(dot(v_tx, tx_boresight_curr), 0)^2;
                    pat_rx = max(dot(v_rx, rx_boresight_curr), 0)^2;
                    pattern_factor = pat_tx * pat_rx;
                    
                    % Wall Penetration Physical Effect Modeling for static background
                    d_tx_wall = 0; tau_tx_wall_delay = 0; trans_loss_tx = 1.0; gamma_tx_eff = wall_gamma;
                    d_rx_wall = 0; tau_rx_wall_delay = 0; trans_loss_rx = 1.0; gamma_rx_eff = wall_gamma;
                    wall_loss_factor = 1.0;

                    if Simulation_Params.enable_wall
                        [d_tx_wall, tau_tx_wall_delay, trans_loss_tx, gamma_tx_eff] = compute_wall_path_terms(tx_pos_curr, target_pos, Simulation_Params.wall_center, Simulation_Params.wall_dimensions, v_wall, c, eps_r);
                        [d_rx_wall, tau_rx_wall_delay, trans_loss_rx, gamma_rx_eff] = compute_wall_path_terms(rx_pos_curr, target_pos, Simulation_Params.wall_center, Simulation_Params.wall_dimensions, v_wall, c, eps_r);
                        wall_loss_factor = trans_loss_tx * trans_loss_rx * exp(-wall_alpha * (d_tx_wall + d_rx_wall));
                    end

                    tau = R_total / c + tau_tx_wall_delay + tau_rx_wall_delay;
                    amp = base_amp_factor * sqrt(current_rcs_val) / (R_tx * R_rx + 1e-6) * wall_loss_factor * pattern_factor;
                    phase = 2*pi*(fc*tau + K*tau*t_fast - 0.5*K*tau^2) + obj_random_phases(p_idx);
                    signal_static_ch = signal_static_ch + amp .* exp(1i * phase);
                    
                    % Precompute multipath ground bounce and complex wall multipath conditionally strictly for static bounds natively
                    if Simulation_Params.enable_multipath
                        gamma_gnd = -0.3;
                        target_pos_mp = [target_pos(1), target_pos(2), -target_pos(3)];
                        R_tx_mp = norm(target_pos_mp - tx_pos_curr);
                        R_rx_mp = norm(target_pos_mp - rx_pos_curr);
                        
                        d_tx_wall_mp = 0; tau_tx_mp_wall_delay = 0; trans_loss_tx_mp = 1.0;
                        d_rx_wall_mp = 0; tau_rx_mp_wall_delay = 0; trans_loss_rx_mp = 1.0;

                        if Simulation_Params.enable_wall
                            [d_tx_wall_mp, tau_tx_mp_wall_delay, trans_loss_tx_mp] = compute_wall_path_terms(tx_pos_curr, target_pos_mp, Simulation_Params.wall_center, Simulation_Params.wall_dimensions, v_wall, c, eps_r);
                            [d_rx_wall_mp, tau_rx_mp_wall_delay, trans_loss_rx_mp] = compute_wall_path_terms(rx_pos_curr, target_pos_mp, Simulation_Params.wall_center, Simulation_Params.wall_dimensions, v_wall, c, eps_r);
                        end

                        v_tx_mpA = (target_pos_mp - tx_pos_curr) / (R_tx_mp + 1e-6);
                        pat_tx_mpA = max(dot(v_tx_mpA, tx_boresight_curr), 0)^2;
                        pattern_factor_A = pat_tx_mpA * pat_rx;
                        tau_mpA = (R_tx_mp + R_rx) / c + tau_tx_mp_wall_delay + tau_rx_wall_delay;
                        amp_loss_A = trans_loss_tx_mp * trans_loss_rx * exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall));
                        amp_mpA = gamma_gnd * base_amp_factor * sqrt(current_rcs_val) / (R_tx_mp * R_rx + 1e-6) * amp_loss_A * pattern_factor_A; 
                        phase_mpA = 2*pi*(fc*tau_mpA + K*tau_mpA*t_fast - 0.5*K*tau_mpA^2) + obj_random_phases(p_idx);
                        signal_static_ch = signal_static_ch + amp_mpA .* exp(1i * phase_mpA);
                        
                        v_rx_mpB = (target_pos_mp - rx_pos_curr) / (R_rx_mp + 1e-6);
                        pat_rx_mpB = max(dot(v_rx_mpB, rx_boresight_curr), 0)^2;
                        pattern_factor_B = pat_tx * pat_rx_mpB;
                        tau_mpB = (R_tx + R_rx_mp) / c + tau_tx_wall_delay + tau_rx_mp_wall_delay;
                        amp_loss_B = trans_loss_tx * trans_loss_rx_mp * exp(-wall_alpha * (d_tx_wall + d_rx_wall_mp));
                        amp_mpB = gamma_gnd * base_amp_factor * sqrt(current_rcs_val) / (R_tx * R_rx_mp + 1e-6) * amp_loss_B * pattern_factor_B; 
                        phase_mpB = 2*pi*(fc*tau_mpB + K*tau_mpB*t_fast - 0.5*K*tau_mpB^2) + obj_random_phases(p_idx);
                        signal_static_ch = signal_static_ch + amp_mpB .* exp(1i * phase_mpB);
                        
                        pattern_factor_C = pat_tx_mpA * pat_rx_mpB;
                        tau_mpC = (R_tx_mp + R_rx_mp) / c + tau_tx_mp_wall_delay + tau_rx_mp_wall_delay;
                        amp_loss_C = trans_loss_tx_mp * trans_loss_rx_mp * exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall_mp));
                        amp_mpC = (gamma_gnd^2) * base_amp_factor * sqrt(current_rcs_val) / (R_tx_mp * R_rx_mp + 1e-6) * amp_loss_C * pattern_factor_C; 
                        phase_mpC = 2*pi*(fc*tau_mpC + K*tau_mpC*t_fast - 0.5*K*tau_mpC^2) + obj_random_phases(p_idx);
                        signal_static_ch = signal_static_ch + amp_mpC .* exp(1i * phase_mpC);
                        
                        if Simulation_Params.enable_wall
                            if target_pos(2) > wall_y_max
                                target_wall_face = wall_y_max;
                            else
                                target_wall_face = wall_y_min;
                            end
                            
                            target_pos_wall_mp = target_pos;
                            target_pos_wall_mp(2) = 2 * target_wall_face - target_pos(2);
                            R_tx_wall_mp = norm(target_pos_wall_mp - tx_pos_curr);
                            R_rx_wall_mp = norm(target_pos_wall_mp - rx_pos_curr);
                            
                            if is_reflection_path_valid(tx_pos_curr, target_pos_wall_mp, target_wall_face, Simulation_Params.wall_center, Simulation_Params.wall_dimensions)
                                v_tx_wall_mp = (target_pos_wall_mp - tx_pos_curr) / (R_tx_wall_mp + 1e-6);
                                pat_tx_wall_mp = max(dot(v_tx_wall_mp, tx_boresight_curr), 0)^2;
                                pattern_factor_2A = pat_tx_wall_mp * pat_rx;
                                tau_mp2A = (R_tx_wall_mp + R_rx) / c + tau_rx_wall_delay; 
                                amp_mp2A = wall_gamma * base_amp_factor * sqrt(current_rcs_val) / (R_tx_wall_mp * R_rx + 1e-6) * trans_loss_rx * exp(-wall_alpha * d_rx_wall) * pattern_factor_2A;
                                phase_mp2A = 2*pi*(fc*tau_mp2A + K*tau_mp2A*t_fast - 0.5*K*tau_mp2A^2) + obj_random_phases(p_idx);
                                signal_static_ch = signal_static_ch + amp_mp2A .* exp(1i * phase_mp2A);
                            end
                            
                            if is_reflection_path_valid(rx_pos_curr, target_pos_wall_mp, target_wall_face, Simulation_Params.wall_center, Simulation_Params.wall_dimensions)
                                v_rx_wall_mp = (target_pos_wall_mp - rx_pos_curr) / (R_rx_wall_mp + 1e-6);
                                pat_rx_wall_mp = max(dot(v_rx_wall_mp, rx_boresight_curr), 0)^2;
                                pattern_factor_2B = pat_tx * pat_rx_wall_mp;
                                tau_mp2B = (R_tx + R_rx_wall_mp) / c + tau_tx_wall_delay;
                                amp_mp2B = wall_gamma * base_amp_factor * sqrt(current_rcs_val) / (R_tx * R_rx_wall_mp + 1e-6) * trans_loss_tx * exp(-wall_alpha * d_tx_wall) * pattern_factor_2B;
                                phase_mp2B = 2*pi*(fc*tau_mp2B + K*tau_mp2B*t_fast - 0.5*K*tau_mp2B^2) + obj_random_phases(p_idx);
                                signal_static_ch = signal_static_ch + amp_mp2B .* exp(1i * phase_mp2B);
                            end
                            
                            if (d_tx_wall > 0 || d_rx_wall > 0)
                                max_ringing_order = 5; 
                                effective_gamma_rt = min(max([abs(gamma_tx_eff), abs(gamma_rx_eff), abs(wall_gamma)]) * 2.0, 0.85); 
                                for m_ring = 1:max_ringing_order
                                    d_internal_bounce = m_ring * 2 * wall_y_thickness;
                                    tau_mp3 = tau + d_internal_bounce / v_wall;
                                    amp_mp3 = amp * (effective_gamma_rt^m_ring) * exp(-wall_alpha * d_internal_bounce); 
                                    phase_mp3 = 2*pi*(fc*tau_mp3 + K*tau_mp3*t_fast - 0.5*K*tau_mp3^2) + obj_random_phases(p_idx);
                                    signal_static_ch = signal_static_ch + amp_mp3 .* exp(1i * phase_mp3);
                                end
                            end
                            
                            if target_pos(2) < wall_y_min && tx_pos_curr(2) > wall_y_max
                                inner_wall_face = wall_y_min;
                                is_valid_twt = true;
                            elseif target_pos(2) > wall_y_max && tx_pos_curr(2) < wall_y_min
                                inner_wall_face = wall_y_max;
                                is_valid_twt = true;
                            else
                                is_valid_twt = false; 
                            end
                            
                            if is_valid_twt
                                D_tw = abs(target_pos(2) - inner_wall_face);
                                if D_tw > 0.1
                                    tau_mp4 = tau + 2 * D_tw / c;
                                    amp_mp4 = amp * abs(wall_gamma) * (R_total / (R_total + 2 * D_tw)) * 0.5;
                                    phase_mp4 = 2*pi*(fc*tau_mp4 + K*tau_mp4*t_fast - 0.5*K*tau_mp4^2) + obj_random_phases(p_idx);
                                    signal_static_ch = signal_static_ch + amp_mp4 .* exp(1i * phase_mp4);
                                    
                                    tau_mp5 = tau + 4 * D_tw / c;
                                    amp_mp5 = amp * (abs(wall_gamma)^2) * (R_total / (R_total + 4 * D_tw)) * 0.25;
                                    phase_mp5 = 2*pi*(fc*tau_mp5 + K*tau_mp5*t_fast - 0.5*K*tau_mp5^2) + obj_random_phases(p_idx);
                                    signal_static_ch = signal_static_ch + amp_mp5 .* exp(1i * phase_mp5);
                                end
                            end
                        end
                    end
                end
            end
            static_echo_matrix(:, ch) = signal_static_ch;
        end
    end

    % Initialize the raw echo matrix containing all fast and slow time domain samples mapped alongside multiple channels natively
    raw_echo = complex(zeros(N_s, num_pulses, num_channels)); % Preallocate as complex array preventing implicit memory reallocations during pulse injection securely evaluating capacities
    
    % Generate intrinsic random scattering phases strictly resolving uniformly over dimensions circumventing false coherent summations structurally
    rng(Random_Seed); % Make phases reproducible across simulation runs
    joint_random_phases = 2 * pi * rand(num_persons, 22);

    % Flatten moving targets to eliminate deep interpreter loops accelerating massive multi target evaluations
    num_M = num_persons * 22;
    target_pos_flat = zeros(num_pulses, num_M, 3);
    target_rcs_flat = zeros(1, num_M);
    target_phase_flat = zeros(1, num_M);
    
    idx = 1;
    for i_p = 1:num_persons
        for j = 1:22
            target_pos_flat(:, idx, 1) = motion_radar_all(:, i_p, j, 1);
            target_pos_flat(:, idx, 2) = motion_radar_all(:, i_p, j, 3);
            target_pos_flat(:, idx, 3) = motion_radar_all(:, i_p, j, 2);
            target_rcs_flat(idx) = Simulation_Params.persons(i_p).Scaled_RCS(j);
            target_phase_flat(idx) = joint_random_phases(i_p, j);
            idx = idx + 1;
        end
    end
    
    N_total = num_pulses * num_M;
    target_pos_flat = reshape(target_pos_flat, N_total, 3);
    target_rcs_flat = repmat(target_rcs_flat, num_pulses, 1);
    target_rcs_flat = target_rcs_flat(:);
    target_phase_flat = repmat(target_phase_flat, num_pulses, 1);
    target_phase_flat = target_phase_flat(:);

    % Removed the GUI waitbar to maintain the immersive agent console output
    agent_think(sprintf("Calculating multi-target MIMO bistatic radar equations linearly propagating signals over %d spatial channels. Advanced matrix vectorization engaged.", num_channels));
    
    for ch = 1:num_channels
        
        % Map channel linear index reliably tracking TX RX combination iteratively safely resolving spatial locations
        tx_idx = ceil(ch / num_rx);
        rx_idx = mod(ch - 1, num_rx) + 1;
        tx_pos_curr = tx_pos_list(tx_idx, :);
        rx_pos_curr = rx_pos_list(rx_idx, :);
        tx_boresight_curr = tx_boresight_list(tx_idx, :);
        rx_boresight_curr = rx_boresight_list(rx_idx, :);
        
        % Vectorized Path Tracking Arrays collecting temporal matrices structurally
        Path_Tau = {};
        Path_Amp = {};
        Path_Phase = {};

        % --- Component 0: Base Direct Path ---
        target_vec_tx = target_pos_flat - tx_pos_curr;
        R_tx = sqrt(sum(target_vec_tx.^2, 2));
        target_vec_rx = target_pos_flat - rx_pos_curr;
        R_rx = sqrt(sum(target_vec_rx.^2, 2));
        R_total = R_tx + R_rx;

        v_tx = target_vec_tx ./ (R_tx + 1e-6);
        v_rx = target_vec_rx ./ (R_rx + 1e-6);
        pat_tx = max(sum(v_tx .* tx_boresight_curr, 2), 0).^2;
        pat_rx = max(sum(v_rx .* rx_boresight_curr, 2), 0).^2;
        pattern_factor = pat_tx .* pat_rx;

        d_tx_wall = zeros(N_total, 1); tau_tx_wall_delay = zeros(N_total, 1); trans_loss_tx = ones(N_total, 1); gamma_tx_eff = repmat(wall_gamma, N_total, 1);
        d_rx_wall = zeros(N_total, 1); tau_rx_wall_delay = zeros(N_total, 1); trans_loss_rx = ones(N_total, 1); gamma_rx_eff = repmat(wall_gamma, N_total, 1);
        wall_loss_factor = ones(N_total, 1);

        if Simulation_Params.enable_wall
            [d_tx_wall, tau_tx_wall_delay, trans_loss_tx, gamma_tx_eff] = compute_wall_path_terms_vec(tx_pos_curr, target_pos_flat, Simulation_Params.wall_center, Simulation_Params.wall_dimensions, v_wall, c, eps_r, wall_gamma);
            [d_rx_wall, tau_rx_wall_delay, trans_loss_rx, gamma_rx_eff] = compute_wall_path_terms_vec(rx_pos_curr, target_pos_flat, Simulation_Params.wall_center, Simulation_Params.wall_dimensions, v_wall, c, eps_r, wall_gamma);
            wall_loss_factor = trans_loss_tx .* trans_loss_rx .* exp(-wall_alpha * (d_tx_wall + d_rx_wall));
        end

        tau_base = R_total / c + tau_tx_wall_delay + tau_rx_wall_delay;
        amp_base = base_amp_factor * sqrt(target_rcs_flat) ./ (R_tx .* R_rx + 1e-6) .* wall_loss_factor .* pattern_factor;

        Path_Tau{end+1} = reshape(tau_base, num_pulses, num_M);
        Path_Amp{end+1} = reshape(amp_base, num_pulses, num_M);
        Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

        if Simulation_Params.enable_multipath
            % Component 1: Multi-Path Ground Modeling
            gamma_gnd = -0.3;
            target_pos_mp = target_pos_flat;
            target_pos_mp(:, 3) = -target_pos_flat(:, 3);
            
            R_tx_mp = sqrt(sum((target_pos_mp - tx_pos_curr).^2, 2));
            R_rx_mp = sqrt(sum((target_pos_mp - rx_pos_curr).^2, 2));
            
            d_tx_wall_mp = zeros(N_total, 1); tau_tx_mp_wall_delay = zeros(N_total, 1); trans_loss_tx_mp = ones(N_total, 1);
            d_rx_wall_mp = zeros(N_total, 1); tau_rx_mp_wall_delay = zeros(N_total, 1); trans_loss_rx_mp = ones(N_total, 1);
            
            if Simulation_Params.enable_wall
                [d_tx_wall_mp, tau_tx_mp_wall_delay, trans_loss_tx_mp, ~] = compute_wall_path_terms_vec(tx_pos_curr, target_pos_mp, Simulation_Params.wall_center, Simulation_Params.wall_dimensions, v_wall, c, eps_r, wall_gamma);
                [d_rx_wall_mp, tau_rx_mp_wall_delay, trans_loss_rx_mp, ~] = compute_wall_path_terms_vec(rx_pos_curr, target_pos_mp, Simulation_Params.wall_center, Simulation_Params.wall_dimensions, v_wall, c, eps_r, wall_gamma);
            end

            v_tx_mpA = (target_pos_mp - tx_pos_curr) ./ (R_tx_mp + 1e-6);
            pat_tx_mpA = max(sum(v_tx_mpA .* tx_boresight_curr, 2), 0).^2;
            pattern_factor_A = pat_tx_mpA .* pat_rx;
            
            tau_mpA = (R_tx_mp + R_rx) / c + tau_tx_mp_wall_delay + tau_rx_wall_delay;
            amp_loss_A = trans_loss_tx_mp .* trans_loss_rx .* exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall));
            amp_mpA = gamma_gnd * base_amp_factor * sqrt(target_rcs_flat) ./ (R_tx_mp .* R_rx + 1e-6) .* amp_loss_A .* pattern_factor_A; 
            
            Path_Tau{end+1} = reshape(tau_mpA, num_pulses, num_M);
            Path_Amp{end+1} = reshape(amp_mpA, num_pulses, num_M);
            Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

            % Route 1.2: TX -> Target -> Ground -> RX
            v_rx_mpB = (target_pos_mp - rx_pos_curr) ./ (R_rx_mp + 1e-6);
            pat_rx_mpB = max(sum(v_rx_mpB .* rx_boresight_curr, 2), 0).^2;
            pattern_factor_B = pat_tx .* pat_rx_mpB;
            
            tau_mpB = (R_tx + R_rx_mp) / c + tau_tx_wall_delay + tau_rx_mp_wall_delay;
            amp_loss_B = trans_loss_tx .* trans_loss_rx_mp .* exp(-wall_alpha * (d_tx_wall + d_rx_wall_mp));
            amp_mpB = gamma_gnd * base_amp_factor * sqrt(target_rcs_flat) ./ (R_tx .* R_rx_mp + 1e-6) .* amp_loss_B .* pattern_factor_B; 
            
            Path_Tau{end+1} = reshape(tau_mpB, num_pulses, num_M);
            Path_Amp{end+1} = reshape(amp_mpB, num_pulses, num_M);
            Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

            % Route 1.3: TX -> Ground -> Target -> Ground -> RX
            pattern_factor_C = pat_tx_mpA .* pat_rx_mpB;
            tau_mpC = (R_tx_mp + R_rx_mp) / c + tau_tx_mp_wall_delay + tau_rx_mp_wall_delay;
            amp_loss_C = trans_loss_tx_mp .* trans_loss_rx_mp .* exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall_mp));
            amp_mpC = (gamma_gnd^2) * base_amp_factor * sqrt(target_rcs_flat) ./ (R_tx_mp .* R_rx_mp + 1e-6) .* amp_loss_C .* pattern_factor_C; 
            
            Path_Tau{end+1} = reshape(tau_mpC, num_pulses, num_M);
            Path_Amp{end+1} = reshape(amp_mpC, num_pulses, num_M);
            Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);
            
            % Components 2, 3 & 4: Wall Structural Multipath Signatures
            if Simulation_Params.enable_wall
                target_wall_face = zeros(N_total, 1);
                idx_top = target_pos_flat(:, 2) > wall_y_max;
                target_wall_face(idx_top) = wall_y_max;
                target_wall_face(~idx_top) = wall_y_min;
                
                % Component 2: Exterior bounce modeling utilizing strict Bistatic Image Theory Geometry
                target_pos_wall_mp = target_pos_flat;
                target_pos_wall_mp(:, 2) = 2 * target_wall_face - target_pos_flat(:, 2);
                
                R_tx_wall_mp = sqrt(sum((target_pos_wall_mp - tx_pos_curr).^2, 2));
                R_rx_wall_mp = sqrt(sum((target_pos_wall_mp - rx_pos_curr).^2, 2));
                
                % Path 2A: TX -> Wall -> Target -> RX
                valid_2A = is_reflection_path_valid_vec(tx_pos_curr, target_pos_wall_mp, target_wall_face, Simulation_Params.wall_center, Simulation_Params.wall_dimensions);
                v_tx_wall_mp = (target_pos_wall_mp - tx_pos_curr) ./ (R_tx_wall_mp + 1e-6);
                pat_tx_wall_mp = max(sum(v_tx_wall_mp .* tx_boresight_curr, 2), 0).^2;
                pattern_factor_2A = pat_tx_wall_mp .* pat_rx;
                
                tau_mp2A = (R_tx_wall_mp + R_rx) / c + tau_rx_wall_delay;
                amp_mp2A = zeros(N_total, 1);
                amp_mp2A(valid_2A) = wall_gamma * base_amp_factor * sqrt(target_rcs_flat(valid_2A)) ./ (R_tx_wall_mp(valid_2A) .* R_rx(valid_2A) + 1e-6) .* trans_loss_rx(valid_2A) .* exp(-wall_alpha * d_rx_wall(valid_2A)) .* pattern_factor_2A(valid_2A);
                
                Path_Tau{end+1} = reshape(tau_mp2A, num_pulses, num_M);
                Path_Amp{end+1} = reshape(amp_mp2A, num_pulses, num_M);
                Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

                % Path 2B: TX -> Target -> Wall -> RX
                valid_2B = is_reflection_path_valid_vec(rx_pos_curr, target_pos_wall_mp, target_wall_face, Simulation_Params.wall_center, Simulation_Params.wall_dimensions);
                v_rx_wall_mp = (target_pos_wall_mp - rx_pos_curr) ./ (R_rx_wall_mp + 1e-6);
                pat_rx_wall_mp = max(sum(v_rx_wall_mp .* rx_boresight_curr, 2), 0).^2;
                pattern_factor_2B = pat_tx .* pat_rx_wall_mp;
                
                tau_mp2B = (R_tx + R_rx_wall_mp) / c + tau_tx_wall_delay;
                amp_mp2B = zeros(N_total, 1);
                amp_mp2B(valid_2B) = wall_gamma * base_amp_factor * sqrt(target_rcs_flat(valid_2B)) ./ (R_tx(valid_2B) .* R_rx_wall_mp(valid_2B) + 1e-6) .* trans_loss_tx(valid_2B) .* exp(-wall_alpha * d_tx_wall(valid_2B)) .* pattern_factor_2B(valid_2B);
                
                Path_Tau{end+1} = reshape(tau_mp2B, num_pulses, num_M);
                Path_Amp{end+1} = reshape(amp_mp2B, num_pulses, num_M);
                Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

                % Component 3: Upgraded to Multi-Order Internal Wall Reverberation
                valid_ring = (d_tx_wall > 0 | d_rx_wall > 0);
                if any(valid_ring)
                    max_ringing_order = 5;
                    effective_gamma_rt = min(max([abs(gamma_tx_eff(valid_ring)), abs(gamma_rx_eff(valid_ring)), repmat(abs(wall_gamma), sum(valid_ring), 1)], [], 2) * 2.0, 0.85);
                    
                    for m_ring = 1:max_ringing_order
                        d_internal_bounce = m_ring * 2 * wall_y_thickness;
                        tau_mp3 = zeros(N_total, 1);
                        tau_mp3(valid_ring) = tau_base(valid_ring) + d_internal_bounce / v_wall;
                        amp_mp3 = zeros(N_total, 1);
                        amp_mp3(valid_ring) = amp_base(valid_ring) .* (effective_gamma_rt.^m_ring) .* exp(-wall_alpha * d_internal_bounce);
                        
                        Path_Tau{end+1} = reshape(tau_mp3, num_pulses, num_M);
                        Path_Amp{end+1} = reshape(amp_mp3, num_pulses, num_M);
                        Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);
                    end
                end

                % Component 4: Target-Wall Room Reverberation
                inner_wall_face = zeros(N_total, 1);
                is_valid_twt = false(N_total, 1);
                
                idx_cond1 = target_pos_flat(:, 2) < wall_y_min & tx_pos_curr(2) > wall_y_max;
                inner_wall_face(idx_cond1) = wall_y_min;
                is_valid_twt(idx_cond1) = true;
                
                idx_cond2 = target_pos_flat(:, 2) > wall_y_max & tx_pos_curr(2) < wall_y_min;
                inner_wall_face(idx_cond2) = wall_y_max;
                is_valid_twt(idx_cond2) = true;
                
                D_tw = abs(target_pos_flat(:, 2) - inner_wall_face);
                valid_4 = is_valid_twt & (D_tw > 0.1);
                
                if any(valid_4)
                    % Path 4A: 1st-Order Reverberation (Target -> Wall -> Target)
                    tau_mp4 = zeros(N_total, 1);
                    tau_mp4(valid_4) = tau_base(valid_4) + 2 * D_tw(valid_4) / c;
                    amp_mp4 = zeros(N_total, 1);
                    amp_mp4(valid_4) = amp_base(valid_4) .* abs(wall_gamma) .* (R_total(valid_4) ./ (R_total(valid_4) + 2 * D_tw(valid_4))) * 0.5;
                    
                    Path_Tau{end+1} = reshape(tau_mp4, num_pulses, num_M);
                    Path_Amp{end+1} = reshape(amp_mp4, num_pulses, num_M);
                    Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);
                    
                    % Path 4B: 2nd-Order Reverberation (Target -> Wall -> Target -> Wall -> Target)
                    tau_mp5 = zeros(N_total, 1);
                    tau_mp5(valid_4) = tau_base(valid_4) + 4 * D_tw(valid_4) / c;
                    amp_mp5 = zeros(N_total, 1);
                    amp_mp5(valid_4) = amp_base(valid_4) .* (abs(wall_gamma)^2) .* (R_total(valid_4) ./ (R_total(valid_4) + 4 * D_tw(valid_4))) * 0.25;
                    
                    Path_Tau{end+1} = reshape(tau_mp5, num_pulses, num_M);
                    Path_Amp{end+1} = reshape(amp_mp5, num_pulses, num_M);
                    Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);
                end
            end
        end

        % Precalculate Antenna Direct Coupling Leakage Physics once per channel securely
        R_leakage = max(norm(tx_pos_curr - rx_pos_curr), 1e-3); % Add minimum near-field threshold strictly preventing nonphysical division by absolute zero singularity scaling
        tau_leakage = R_leakage / c;
        if isfield(Simulation_Params, 'antenna_isolation')
            iso_lin = 10^(-Simulation_Params.antenna_isolation / 20);
        else
            iso_lin = 10^(-40 / 20);
        end
        amp_leakage = (lambda * G_lin) / (4 * pi * R_leakage) * iso_lin;
        phase_leakage = 2*pi*(fc*tau_leakage + K*tau_leakage*t_fast - 0.5*K*tau_leakage^2);

        % Combine coherent multi-path phase signals rigorously inside bounded vector additions natively
        num_paths = length(Path_Tau);
        for p = 1:num_pulses
            signal_p = complex(zeros(N_s, 1));
            signal_p = signal_p + amp_leakage .* exp(1i * phase_leakage) + static_echo_matrix(:, ch);
            
            for path_idx = 1:num_paths
                tau_p = Path_Tau{path_idx}(p, :);
                amp_p = Path_Amp{path_idx}(p, :);
                phase_flat_p = Path_Phase{path_idx}(p, :);
                
                % Process mathematically active structural bounds bypassing null elements cleanly accelerating performance
                valid = amp_p > 1e-12;
                if any(valid)
                    tau_v = tau_p(valid);
                    amp_v = amp_p(valid);
                    phase_v = phase_flat_p(valid);
                    
                    % Calculate the beat signal phase vector and stack the complex exponential wave tracking isolated phases
                    phase_mat = 2*pi*(fc*tau_v + K*t_fast*tau_v - 0.5*K*tau_v.^2) + phase_v;
                    signal_p = signal_p + sum(amp_v .* exp(1i * phase_mat), 2);
                end
            end
            % Assign the synthesized pulse to the entire 3D raw echo matrix bridging multi coherent channels inherently
            raw_echo(:, p, ch) = signal_p;
        end
    end
    
    % Accommodate the FFT processing gain over N_s enabling rigorous physics level SNR simulations rendering visually consistent noise floor
    ref_distance = 2.0;
    ref_amp = (lambda * G_lin * 1.0) / ((4*pi)^1.5 * ref_distance^2);
    ref_power = ref_amp^2;
    noise_power = (ref_power / (10^(Simulation_Params.SNR / 10))) * N_s;
    
    % Append independent noise dynamically spanning entire 3D tensor cleanly allocating statistical variances uniquely
    noise_matrix = sqrt(noise_power/2) * (randn(size(raw_echo)) + 1i * randn(size(raw_echo)));
    
    raw_echo = raw_echo + noise_matrix;
    save(fullfile(Save_Dir, 'Raw_Echo.mat'), 'raw_echo', '-v7.3');
    
    %% Signal Processing: RTM & DTM Generation Across Multiple Channels
    % Replace default print with agent thinking stream
    agent_think("Executing discrete numerical fourier transforms upon the simulated MIMO echoes resolving spatial range coordinates natively propagating variables dynamically.");
    
    % Apply structural windowing function critically reducing fast time sinc range sidelobes strictly mapping across elements cleanly
    fast_time_window = hamming(N_s);
    raw_echo_windowed = raw_echo .* fast_time_window;
    
    % Apply fast fourier transform along the fast time dimension spanning channels inherently
    range_fft = fft(raw_echo_windowed, N_s, 1);
    
    % Keep only the positive frequencies to construct the complex range time matrix
    RTM_complex_all = range_fft(1:floor(N_s/2), :, :);         
    
    % Calculate the corresponding physical range axis mapped to the beat frequencies
    f_beat = (0:floor(N_s/2)-1)' * (fs / N_s);
    range_axis = f_beat * c / (2 * K);

    % Define a realistic maximum boundary for the indoor physical environment
    max_indoor_range = 25.0;                                                
    
    % Filter out the redundant distant range bins accelerating neural network inference
    valid_range_idx = range_axis <= max_indoor_range;
    RTM_complex_all = RTM_complex_all(valid_range_idx, :, :);
    range_axis = range_axis(valid_range_idx);
    
    % Clear immense memory blocks releasing gigabytes of system resources ensuring simulation stability preventing OOM
    clear range_fft raw_echo_windowed raw_echo noise_matrix;
    
    % Apply Moving Target Indication (MTI) subtracting static constant targets (such as walls, structural objects and DC leakage component) resolving dimension 3 cleanly
    agent_think("Applying Moving Target Indication (MTI) universally subtracting exact static backgrounds isolating purely kinetic micro-Doppler signatures seamlessly.");
    RTM_complex_all = RTM_complex_all - mean(RTM_complex_all, 2);
    
    % Embed hardware Sensitivity Time Control (STC) overcoming long range dramatic amplitude collapsing behaviors via R^2 weight natively tracking
    STC_Weights = (range_axis.^2) + 1e-3;
    [~, ref_idx] = min(abs(range_axis - 2.0));
    STC_Weights = STC_Weights / STC_Weights(max(1, ref_idx)); 
    RTM_complex_all = RTM_complex_all .* STC_Weights;

    % Compute the linear forms extracting strictly isolated configurations capturing First Channel mapping and unified Channel Sum integrations respectively
    RTM_Mag_Lin_All = abs(RTM_complex_all);
    RTM_Mag_Lin_First = RTM_Mag_Lin_All(:, :, 1);
    RTM_Mag_Lin_Sum = sum(RTM_Mag_Lin_All, 3) / num_channels; % Incoherent summation explicitly solving spatial fading patterns realistically
    
    % Dynamic continuous tracking capturing unconstrained multi motions bounding valid spatial locations perfectly evading signal truncations using global Sum limits
    range_variance = var(RTM_Mag_Lin_Sum, 0, 2);
    active_threshold = max(range_variance) * 0.05; 
    active_bins = find(range_variance > active_threshold);
    
    if isempty(active_bins)
        [~, max_idx] = max(mean(RTM_Mag_Lin_Sum, 2));
        active_bins = max_idx;
    end
    
    % Establish dynamic plotting bounds dynamically framing actions fully resolving complex traces natively overlapping
    plot_r_min = max(0, range_axis(active_bins(1)) - 2.0);
    plot_r_max = min(max_indoor_range, range_axis(active_bins(end)) + 8.0);
    
    % Preallocate time and frequency parameters extracting proper dimensions via a single spectrogram call securely
    n_window = min(num_pulses, max(round(Simulation_Params.stft_window_seconds * PRF), 8));
    if mod(n_window, 2) == 0 && n_window > 1
        n_window = n_window - 1;
    end
    n_window = max(n_window, min(num_pulses, 5));
    n_overlap = min(round(n_window * Simulation_Params.stft_overlap_ratio), max(n_window - 1, 0));
    nfft_stft = min(4096, max(256, 2^nextpow2(max(n_window, 32))));
    [~, F_doppler, time_stft] = spectrogram(RTM_complex_all(active_bins(1), :, 1), hamming(n_window), n_overlap, nfft_stft, PRF, 'centered');
    
    % Allocate spatial temporal energetic arrays extracting First Channel isolated properties and inclusive global mappings securely
    DTM_Energy_First = zeros(length(F_doppler), length(time_stft));
    DTM_Energy_Sum = zeros(length(F_doppler), length(time_stft));
    
    % Incoherent STFT Integration: Summing magnitude-squared energy globally circumventing structural truncation tracking arrays efficiently incorporating MIMO bounds
    for ch = 1:num_channels
        DTM_Energy_Ch = zeros(length(F_doppler), length(time_stft));
        for b = 1:length(active_bins)
            bin_idx = active_bins(b);
            [S_bin, ~, ~] = spectrogram(RTM_complex_all(bin_idx, :, ch), hamming(n_window), n_overlap, nfft_stft, PRF, 'centered');
            DTM_Energy_Ch = DTM_Energy_Ch + abs(S_bin).^2;
        end
        if ch == 1
            DTM_Energy_First = DTM_Energy_Ch;
        end
        DTM_Energy_Sum = DTM_Energy_Sum + DTM_Energy_Ch;
    end
    
    % Average cumulative energies strictly evaluating boundaries preventing numerical bounds overflow correctly
    DTM_Energy_Sum = DTM_Energy_Sum / num_channels;
    
    % Re-polarize visual direction aligning target approach behavior conforming to common strictly positive Doppler expectations 
    DTM_Mag_Lin_First = sqrt(flipud(DTM_Energy_First));
    DTM_Mag_Lin_Sum = sqrt(flipud(DTM_Energy_Sum));

    %% Magnitude Normalization and Logarithmic Conversion
    % Replace default print with agent thinking stream
    agent_think("Normalizing matrix magnitudes independently and converting to rigorous logarithmic decibel scales representing multiple channel representations seamlessly.");
    
    % Compute standard decibel (dB) logarithmic scaling correctly applying 20log10 securely evaluating thresholds
    RTM_Mag_Log_First = 20*log10(RTM_Mag_Lin_First + 1e-12);
    DTM_Mag_Log_First = 20*log10(DTM_Mag_Lin_First + 1e-12);
    
    RTM_Mag_Log_Sum = 20*log10(RTM_Mag_Lin_Sum + 1e-12);
    DTM_Mag_Log_Sum = 20*log10(DTM_Mag_Lin_Sum + 1e-12);
    
    % Align the upper global ceiling strictly mapping peak responses mapping to pristine 0 dB
    RTM_Mag_Log_First_Norm = RTM_Mag_Log_First - max(RTM_Mag_Log_First(:));
    DTM_Mag_Log_First_Norm = DTM_Mag_Log_First - max(DTM_Mag_Log_First(:));
    
    RTM_Mag_Log_Sum_Norm = RTM_Mag_Log_Sum - max(RTM_Mag_Log_Sum(:));
    DTM_Mag_Log_Sum_Norm = DTM_Mag_Log_Sum - max(DTM_Mag_Log_Sum(:));

    %% Network Denoising
    if Simulation_Params.enable_network
        % Replace default print with agent thinking stream
        agent_think("Connecting the local deep learning denoising framework to systematically eliminate background static while preserving dynamic kinematic MIMO channel traces.");
        try
            % Load the predefined image denoising convolutional neural network
            net = denoisingNetwork('dncnn');
            
            % Cast to single protecting computational memory securing reliable denoise inference executions cleanly
            denoise_matrix = @(Mat) ...
                double(denoiseImage(single((Mat - min(Mat(:))) / (max(Mat(:)) - min(Mat(:)) + 1e-12)), net)) * (max(Mat(:)) - min(Mat(:))) + min(Mat(:));
            
            % Execute the noise removal logic utilizing robust scaled definitions enhancing physical shapes universally across channels
            RTM_Mag_Log_First_Norm = denoise_matrix(RTM_Mag_Log_First_Norm);
            DTM_Mag_Log_First_Norm = denoise_matrix(DTM_Mag_Log_First_Norm);
            
            RTM_Mag_Log_Sum_Norm = denoise_matrix(RTM_Mag_Log_Sum_Norm);
            DTM_Mag_Log_Sum_Norm = denoise_matrix(DTM_Mag_Log_Sum_Norm);
            
        catch ME
            % Replace default warning with agent thinking stream
            agent_think("The deep neural network enhancement mechanism produced an internal operational failure. Terminating the refinement routine to maintain spectral stability.");
        end
    end

    %% Export High Resolution Dataset PNG Images
    % Replace default print with agent thinking stream
    agent_think("Formulating completely distinct First Channel and Channel Sum compressed high fidelity image matrices and exporting to the drive in strict png format natively.");
    
    % Define the fixed logarithmic dynamic range limits for feature enhancement
    log_clim_min = -30;
    log_clim_max = 0;
    
    % Clamp the normalized logarithmic matrices enforcing the specified visual bounds correctly scaling pixel depths linearly
    RTM_First_Log_Clamped = max(min(RTM_Mag_Log_First_Norm, log_clim_max), log_clim_min);
    DTM_First_Log_Clamped = max(min(DTM_Mag_Log_First_Norm, log_clim_max), log_clim_min);
    
    RTM_Sum_Log_Clamped = max(min(RTM_Mag_Log_Sum_Norm, log_clim_max), log_clim_min);
    DTM_Sum_Log_Clamped = max(min(DTM_Mag_Log_Sum_Norm, log_clim_max), log_clim_min);
    
    % Map the custom colormap dynamically replacing generic mappings maintaining consistent high fidelity visual forms universally
    custom_cmap_256 = interp1(linspace(0, 1, size(JoeyBG_Colormap_Flip, 1)), JoeyBG_Colormap_Flip, linspace(0, 1, 256));
    
    % Local nested rendering script correctly formulating identical pipeline definitions recursively mapping four discrete outputs securely
    render_and_save_png = @(Matrix_Log, file_name) ...
        imwrite(imresize(ind2rgb(gray2ind(normalize_log_image_for_export(Matrix_Log, log_clim_min, log_clim_max), 256), custom_cmap_256), [1024, 1024]), fullfile(Save_Dir, file_name));

    % Execute the bounded render processes mapping distinct formats identically natively exporting png extensions
    render_and_save_png(RTM_First_Log_Clamped, 'RTM_First_Channel_Log_1024x1024.png');
    render_and_save_png(DTM_First_Log_Clamped, 'DTM_First_Channel_Log_1024x1024.png');
    render_and_save_png(RTM_Sum_Log_Clamped,   'RTM_Channel_Sum_Log_1024x1024.png');
    render_and_save_png(DTM_Sum_Log_Clamped,   'DTM_Channel_Sum_Log_1024x1024.png');

    %% Plotting RTM and DTM User Interfaces
    % Replace default print with agent thinking stream
    agent_think("Generating comprehensive multithreaded graphical interface components displaying four requested separate structures mapping First Channel and Channel Sum explicitly.");
    
    % Local anonymous function structuring identically scoped generic plot formulations uniformly evaluating limits cleanly
    create_spectrum_figure = @(Title_Str, X_Axis, Y_Axis, Matrix, Pos_Offset, Y_Lims, Y_Label) ...
        local_spectrum_plotter(Title_Str, X_Axis, Y_Axis, Matrix, Pos_Offset, Y_Lims, Y_Label, Font_Name, Font_Size_Basis, Font_Weight_Basis, Font_Size_Title, Font_Weight_Title, Font_Size_Axis, Font_Weight_Axis, JoeyBG_Colormap_Flip);
    
    % Render Figure 1: RTM (First Channel, Log)
    create_spectrum_figure('RTM (First Channel, Log)', t_slow, range_axis, RTM_Mag_Log_First_Norm, [100, 100], [plot_r_min, plot_r_max], 'Range (m)');
    
    % Render Figure 2: DTM (First Channel, Log)
    create_spectrum_figure('DTM (First Channel, Log)', time_stft, F_doppler, DTM_Mag_Log_First_Norm, [150, 150], [-PRF/2, PRF/2], 'Doppler (Hz)');
    
    % Render Figure 3: RTM (Channel Sum, Log)
    create_spectrum_figure('RTM (Channel Sum, Log)', t_slow, range_axis, RTM_Mag_Log_Sum_Norm, [200, 200], [plot_r_min, plot_r_max], 'Range (m)');
    
    % Render Figure 4: DTM (Channel Sum, Log)
    create_spectrum_figure('DTM (Channel Sum, Log)', time_stft, F_doppler, DTM_Mag_Log_Sum_Norm, [250, 250], [-PRF/2, PRF/2], 'Doppler (Hz)');

    %% Visualizing the Motion
    % Replace default print with agent thinking stream
    agent_think("Constructing the primary three dimensional workspace rendering pipeline to project and animate the physical motions and structural scene layouts mapping all active independent antenna clusters natively.");
    
    frames = Simulation_Params.num_frames;

    % Setup the main figure and graphical axes using given formatting arrays
    fig = figure('Name', 'EMDM 3D Motion Visualization (MIMO)', 'Position',[100, 100, 900, 800]);
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
    
    % Bind the colormap mapping correctly for RCS visualization scaling dynamically
    colormap(ax, JoeyBG_Colormap_Flip);
    clim(ax,[0, 1]); 
    
    % Add the colorbar mapping bar to the edge of the visualizer
    cb = colorbar(ax);
    ylabel(cb, 'Normalized Base RCS Value', 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
    
    % Calculate the dynamic spatial boundaries for the three dimensional scene bounding all subjects inherently
    pad = 0.5;
    x_min = inf; x_max = -inf; y_min = inf; y_max = -inf;
    for i_p = 1:num_persons
        curr_X = Simulation_Params.persons(i_p).motion_mat(:, :, 1);
        curr_Y = Simulation_Params.persons(i_p).motion_mat(:, :, 3); 
        x_min = min(x_min, min(curr_X, [], 'all'));
        x_max = max(x_max, max(curr_X, [], 'all'));
        y_min = min(y_min, min(curr_Y, [], 'all'));
        y_max = max(y_max, max(curr_Y, [], 'all'));
    end
    
    % Dynamically evaluate spatial boundaries capturing placed static objects natively preventing out of bounds visually
    if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
        for i_o = 1:length(Simulation_Params.objects)
            sp_matrix = Simulation_Params.objects(i_o).scatter_points;
            if ~isempty(sp_matrix) && size(sp_matrix, 2) >= 3
                x_min = min(x_min, min(sp_matrix(:, 1)) - pad);
                x_max = max(x_max, max(sp_matrix(:, 1)) + pad);
                y_min = min(y_min, min(sp_matrix(:, 2)) - pad);
                y_max = max(y_max, max(sp_matrix(:, 2)) + pad);
            end
        end
    end
    
    % Adjust margins accommodating multi dimensional spatial hardware array boundaries accurately
    x_min = min(x_min, min([tx_pos_list(:, 1); rx_pos_list(:, 1)])) - pad;
    x_max = max(x_max, max([tx_pos_list(:, 1); rx_pos_list(:, 1)])) + pad;
    y_min = min(y_min, min([tx_pos_list(:, 2); rx_pos_list(:, 2)])) - pad;
    y_max = max(y_max, max([tx_pos_list(:, 2); rx_pos_list(:, 2)])) + pad;
    
    z_max = max([max(tx_pos_list(:, 3)), max(rx_pos_list(:, 3)), 2.2]);
    if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
        for i_o = 1:length(Simulation_Params.objects)
            sp_matrix = Simulation_Params.objects(i_o).scatter_points;
            if ~isempty(sp_matrix) && size(sp_matrix, 2) >= 3
                z_max = max(z_max, max(sp_matrix(:, 3)) + pad);
            end
        end
    end

    xlim(ax,[x_min, x_max]);
    ylim(ax,[y_min, y_max]);
    zlim(ax,[0, z_max]);
    
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
         
    % Plot explicitly iterated multiple antenna models visually bounding arrays mapped effectively tracking configurations directly
    for t = 1:num_tx
        if t == 1
            plot3(ax, tx_pos_list(t, 1), tx_pos_list(t, 2), tx_pos_list(t, 3), '^', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(2,:), 'MarkerEdgeColor', 'k', 'DisplayName', 'TX Antenna(s)');
        else
            plot3(ax, tx_pos_list(t, 1), tx_pos_list(t, 2), tx_pos_list(t, 3), '^', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(2,:), 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
        end
    end
    
    for r = 1:num_rx
        if r == 1
            plot3(ax, rx_pos_list(r, 1), rx_pos_list(r, 2), rx_pos_list(r, 3), 'v', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(end-1,:), 'MarkerEdgeColor', 'k', 'DisplayName', 'RX Antenna(s)');
        else
            plot3(ax, rx_pos_list(r, 1), rx_pos_list(r, 2), rx_pos_list(r, 3), 'v', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(end-1,:), 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
        end
    end
    
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

    % Plot the static objects if present accurately evaluating geometrical constraints visually
    if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
        agent_think("Rendering the reconstructed simple static objects mapping complex geometries via dense structural scattering clusters in three dimensional space.");
        
        plot3(ax, nan, nan, nan, 's', 'MarkerFaceColor', JoeyBG_Colormap(9, :), 'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'DisplayName', 'Static Scene Objects');
        for i_o = 1:length(Simulation_Params.objects)
            obj = Simulation_Params.objects(i_o);
            if isfield(obj, 'scatter_points') && size(obj.scatter_points, 2) >= 3
                sp_matrix = obj.scatter_points(:, 1:4);
                
                for i_sp = 1:size(sp_matrix, 1)
                    if size(sp_matrix, 2) >= 4
                        sc_rcs = sp_matrix(i_sp, 4);
                    else
                        sc_rcs = 0.5;
                    end
                    % Scale rendering markers correctly tracking radar physics profiles proportionally
                    sc_size = max(5, min(20, sqrt(sc_rcs) * 12));
                    
                    plot3(ax, sp_matrix(i_sp, 1), sp_matrix(i_sp, 2), sp_matrix(i_sp, 3), 's', ...
                          'MarkerSize', sc_size, 'MarkerFaceColor', JoeyBG_Colormap(9, :), ...
                          'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
                end
                
                % Render semitransparent convex boundaries linking distributed structural points safely wrapping physical models natively
                if size(sp_matrix, 1) >= 4
                    try
                        K_hull = convhull(sp_matrix(:,1), sp_matrix(:,2), sp_matrix(:,3));
                        trisurf(K_hull, sp_matrix(:,1), sp_matrix(:,2), sp_matrix(:,3), 'Parent', ax, 'FaceColor', JoeyBG_Colormap(9,:), 'FaceAlpha', 0.15, 'EdgeColor', JoeyBG_Colormap(9,:), 'EdgeAlpha', 0.3, 'HandleVisibility', 'off');
                    catch
                        % Silently ignore coplanar geometries avoiding crashing rendering arrays smoothly
                    end
                end
            end
        end
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
    
    % Round mapping boundaries safely mapping the baseline colors to radar section indices tracking uniform representations across persons
    rcs_color_indices = round((1 - Normalized_RCS) * (cmap_len - 1)) + 1;
    rcs_color_indices = max(min(rcs_color_indices, cmap_len), 1); 
    
    % Create graphical object array structures to handle massive graphics updating mapping all active profiles
    h_joints_t = gobjects(num_persons, num_joints);
    h_bones_t = gobjects(num_persons, num_bones);
    h_trajs = gobjects(num_persons, num_joints);
    
    % Allocate NaN arrays tracking spatial trailing histories dynamically frame iteratively mapped correctly
    traj_X = cell(num_persons, 1);
    traj_Y = cell(num_persons, 1);
    traj_Z = cell(num_persons, 1);
    
    for i_p = 1:num_persons
        traj_X{i_p} = nan(frames, num_joints);
        traj_Y{i_p} = nan(frames, num_joints);
        traj_Z{i_p} = nan(frames, num_joints);
        
        % Initialize the three dimensional spherical joints and hidden trailing tracks universally
        for j = 1:num_joints
            rcs_idx = rcs_color_indices(j);
            joint_color = JoeyBG_Colormap(rcs_idx, :);
            
            h_joints_t(i_p, j) = hgtransform('Parent', ax);
            surf(sph_x, sph_y, sph_z, 'Parent', h_joints_t(i_p, j), 'FaceColor', joint_color, ...
                 'EdgeColor', 'none', 'HandleVisibility', 'off', ...
                 'SpecularStrength', 0.2, 'DiffuseStrength', 0.8);
                 
            h_trajs(i_p, j) = plot3(ax, nan, nan, nan, 'Color',[joint_color, 0.35], ...
                               'LineWidth', 1.5, 'HandleVisibility', 'off');
        end
        
        % Initialize the three dimensional cylindrical bones blending connected colors accurately mapping individual sizes
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
            
            h_bones_t(i_p, b) = hgtransform('Parent', ax);
            
            surf(cyl_x, cyl_y, cyl_z, bone_cdata, 'Parent', h_bones_t(i_p, b), ...
                 'FaceColor', 'interp', ...
                 'EdgeColor', 'none', 'HandleVisibility', 'off', ...
                 'SpecularStrength', 0.1, 'DiffuseStrength', 0.9);
        end
    end

    % Removed GUI waitbar for 3D visualization and replaced with thinking stream
    agent_think("Synchronizing internal plot rendering loops to animate multi target joints along their mathematically resolved spatial vectors natively rendering distinct sizes.");
    
    % Execute the primary animation loop over all chronological sample frames mapping massive entities correctly
    for f = 1:frames
        if ~isgraphics(ax)
            break; 
        end
        
        for i_p = 1:num_persons
            % Extract the three dimensional coordinates slicing the entire matrix block natively capturing individual frames
            curr_joints = squeeze(Simulation_Params.persons(i_p).motion_mat(f, :, :));
            
            % Remap the standard axis vectors replacing coordinates natively 
            X_data = curr_joints(:, 1);
            Y_data = curr_joints(:, 3); 
            Z_data = curr_joints(:, 2); 
            
            % Record spatial trajectories continually appending location variables safely indexing arrays
            traj_X{i_p}(f, :) = X_data';
            traj_Y{i_p}(f, :) = Y_data';
            traj_Z{i_p}(f, :) = Z_data';
            
            % Update the graphic spatial trailing instances drawing connected polylines dynamically
            for j = 1:num_joints
                h_trajs(i_p, j).XData = traj_X{i_p}(1:f, j);
                h_trajs(i_p, j).YData = traj_Y{i_p}(1:f, j);
                h_trajs(i_p, j).ZData = traj_Z{i_p}(1:f, j);
            end
            
            % Manipulate the spherical transformation blocks updating respective centers tracking distinct physical sizes appropriately
            for j = 1:num_joints
                T = makehgtform('translate',[X_data(j), Y_data(j), Z_data(j)]);
                S = makehgtform('scale', Simulation_Params.persons(i_p).joint_radii(j));
                h_joints_t(i_p, j).Matrix = T * S;
            end
            
            % Reconstruct the intermediate bone transformation blocks updating lengths rotation adapting distinct physical signatures seamlessly
            for b = 1:num_bones
                j1 = kinematic_tree(b, 1);
                j2 = kinematic_tree(b, 2);
                
                p1 =[X_data(j1), Y_data(j1), Z_data(j1)];
                p2 =[X_data(j2), Y_data(j2), Z_data(j2)];
                
                h_bones_t(i_p, b).Matrix = compute_bone_matrix(p1, p2, Simulation_Params.persons(i_p).bone_radius);
            end
        end
        
        drawnow;
        pause(1/Simulation_Params.FPS); 
    end
    
    % Replace default print with agent thinking stream
    agent_think("The entire complex sequence of multi-person mathematical simulation formulations and distinct structural MIMO visual rendering processes has been successfully concluded.");

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

% Local helper routine securely cleaning arithmetic expressions and formatting flaws from LLM outputs natively mapping JSON rules
function cleaned_str = clean_json_math(raw_str)
    cleaned_str = raw_str;
    % Fix stray letters before keys, e.g., A"Refined_Prompt" -> "Refined_Prompt"
    cleaned_str = regexprep(cleaned_str, '([\{\[,]\s*)[a-zA-Z0-9_]+"', '$1"');
    
    % Fix trailing commas before closing brackets
    cleaned_str = regexprep(cleaned_str, ',\s*([\]\}])', '$1');
    
    % Fix boolean capitalization mapping strictly valid standards
    cleaned_str = regexprep(cleaned_str, ':\s*True', ': true', 'ignorecase');
    cleaned_str = regexprep(cleaned_str, ':\s*False', ': false', 'ignorecase');
    
    % Fix mathematical expressions like 1.5 + 0.6 inside the JSON arrays iteratively substituting scalar evaluations
    pattern = '(?<![eEa-zA-Z_\.])([-+]?(?:\d*\.\d+|\d+))\s*([\+\-\*\/])\s*([-+]?(?:\d*\.\d+|\d+))';
    max_iters = 1000;
    iter = 0;
    while iter < max_iters
        [start_idx, end_idx, tokens] = regexp(cleaned_str, pattern, 'start', 'end', 'tokens', 'once');
        if isempty(start_idx)
            break;
        end
        val1 = str2double(tokens{1});
        op = tokens{2};
        val2 = str2double(tokens{3});
        switch op
            case '+'
                res = val1 + val2;
            case '-'
                res = val1 - val2;
            case '*'
                res = val1 * val2;
            case '/'
                res = val1 / val2;
            otherwise
                res = 0;
        end
        if ~isfinite(res)
            res = 0;
        end
        cleaned_str = [cleaned_str(1:start_idx-1), num2str(res, 10), cleaned_str(end_idx+1:end)];
        iter = iter + 1;
    end
end

% Local formulation routine structuring plot properties cleanly circumventing dense redundancies
function local_spectrum_plotter(Title_Str, X_Axis, Y_Axis, Matrix, Pos_Offset, Y_Lims, Y_Label, Font_Name, Font_Size_Basis, Font_Weight_Basis, Font_Size_Title, Font_Weight_Title, Font_Size_Axis, Font_Weight_Axis, Cmap)
    fig_temp = figure('Name', Title_Str, 'Position', [Pos_Offset(1), Pos_Offset(2), 700, 500]);
    ax_temp = axes('Parent', fig_temp);
    imagesc(ax_temp, X_Axis, Y_Axis, Matrix);
    set(ax_temp, 'YDir', 'normal', 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
    colormap(ax_temp, Cmap); colorbar(ax_temp);
    clim(ax_temp, [-30, 0]);
    title(ax_temp, Title_Str, 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
    xlabel(ax_temp, 'Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylabel(ax_temp, Y_Label, 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
    ylim(ax_temp, Y_Lims); 
end

% Local subroutine defining strict robust boundaries extracting variable MIMO antenna properties flawlessly natively 
function pos_matrix = parse_antenna_positions(pos)
    if iscell(pos)
        pos_matrix = [];
        for i = 1:numel(pos)
            item = pos{i};
            if iscell(item)
                item = cell2mat(item);
            end
            pos_matrix = [pos_matrix; item(:)'];
        end
    elseif isnumeric(pos)
        if size(pos, 2) == 3
            pos_matrix = pos;
        elseif size(pos, 1) == 3 && size(pos, 2) == 1
            pos_matrix = pos';
        elseif numel(pos) == 3
            pos_matrix = pos(:)';
        else
            pos_matrix = reshape(pos, 3, [])';
        end
    else
        pos_matrix = [0, 5, 1];
    end
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

% Local helper routine sanitizing the extracted global simulation parameters without removing any features
function Simulation_Params = sanitize_simulation_params(Simulation_Params, Default_Params)
    Simulation_Params.fc = clamp_scalar(get_struct_value(Simulation_Params, 'fc', Default_Params.fc), 1e8, 3e11, Default_Params.fc);
    Simulation_Params.tp = clamp_scalar(get_struct_value(Simulation_Params, 'tp', Default_Params.tp), 1e-6, 1.0, Default_Params.tp);
    Simulation_Params.B = clamp_scalar(get_struct_value(Simulation_Params, 'B', Default_Params.B), 1e6, 20e9, Default_Params.B);
    Simulation_Params.PRF = clamp_scalar(get_struct_value(Simulation_Params, 'PRF', Default_Params.PRF), 1.0, 2e5, Default_Params.PRF);
    Simulation_Params.fs = clamp_scalar(get_struct_value(Simulation_Params, 'fs', Default_Params.fs), 1e3, 2e8, Default_Params.fs);
    Simulation_Params.antenna_gain = clamp_scalar(get_struct_value(Simulation_Params, 'antenna_gain', Default_Params.antenna_gain), -10, 60, Default_Params.antenna_gain);
    Simulation_Params.antenna_isolation = clamp_scalar(get_struct_value(Simulation_Params, 'antenna_isolation', Default_Params.antenna_isolation), 0, 120, Default_Params.antenna_isolation);
    Simulation_Params.SNR = clamp_scalar(get_struct_value(Simulation_Params, 'SNR', Default_Params.SNR), -20, 120, Default_Params.SNR);
    Simulation_Params.stft_window_seconds = clamp_scalar(get_struct_value(Simulation_Params, 'stft_window_seconds', Default_Params.stft_window_seconds), 0.02, 2.0, Default_Params.stft_window_seconds);
    Simulation_Params.stft_overlap_ratio = clamp_scalar(get_struct_value(Simulation_Params, 'stft_overlap_ratio', Default_Params.stft_overlap_ratio), 0.0, 0.95, Default_Params.stft_overlap_ratio);

    Simulation_Params.enable_wall = coerce_boolean(get_struct_value(Simulation_Params, 'enable_wall', Default_Params.enable_wall));
    Simulation_Params.enable_multipath = coerce_boolean(get_struct_value(Simulation_Params, 'enable_multipath', Default_Params.enable_multipath));
    Simulation_Params.enable_network = coerce_boolean(get_struct_value(Simulation_Params, 'enable_network', Default_Params.enable_network));

    Simulation_Params.tx_pos = parse_antenna_positions(get_struct_value(Simulation_Params, 'tx_pos', Default_Params.tx_pos));
    Simulation_Params.rx_pos = parse_antenna_positions(get_struct_value(Simulation_Params, 'rx_pos', Default_Params.rx_pos));
    Simulation_Params.wall_center = ensure_row_vector(get_struct_value(Simulation_Params, 'wall_center', Default_Params.wall_center), Default_Params.wall_center, 3);
    Simulation_Params.wall_dimensions = abs(ensure_row_vector(get_struct_value(Simulation_Params, 'wall_dimensions', Default_Params.wall_dimensions), Default_Params.wall_dimensions, 3));
    Simulation_Params.wall_dimensions = max(Simulation_Params.wall_dimensions, [0.1, 0.02, 0.3]);
    Simulation_Params.wall_epsilon_r = clamp_scalar(get_struct_value(Simulation_Params, 'wall_epsilon_r', Default_Params.wall_epsilon_r), 1.0, 30.0, Default_Params.wall_epsilon_r);
    Simulation_Params.wall_loss_tangent = clamp_scalar(get_struct_value(Simulation_Params, 'wall_loss_tangent', Default_Params.wall_loss_tangent), 0.0, 1.0, Default_Params.wall_loss_tangent);

    if ~isfield(Simulation_Params, 'persons') || isempty(Simulation_Params.persons)
        Simulation_Params.persons = struct('Refined_Prompt', "a person stands naturally and remains still", ...
                                          'start_pos', [0, 0, 0], ...
                                          'start_heading', 0.0, ...
                                          'start_time_delay', 0.0, ...
                                          'height', 1.70, ...
                                          'weight', 70.0);
    end

    for i_p = 1:length(Simulation_Params.persons)
        if ~isfield(Simulation_Params.persons(i_p), 'Refined_Prompt') || strlength(strtrim(string(Simulation_Params.persons(i_p).Refined_Prompt))) == 0
            Simulation_Params.persons(i_p).Refined_Prompt = "a person stands naturally and remains still";
        else
            Simulation_Params.persons(i_p).Refined_Prompt = string(Simulation_Params.persons(i_p).Refined_Prompt);
        end

        Simulation_Params.persons(i_p).start_pos = ensure_row_vector(get_struct_value(Simulation_Params.persons(i_p), 'start_pos', [0, 0, 0]), [0, 0, 0], 3);
        Simulation_Params.persons(i_p).start_heading = mod(clamp_scalar(get_struct_value(Simulation_Params.persons(i_p), 'start_heading', 0.0), -3600, 3600, 0.0), 360);
        Simulation_Params.persons(i_p).start_time_delay = max(0.0, double(get_struct_value(Simulation_Params.persons(i_p), 'start_time_delay', 0.0)));
        Simulation_Params.persons(i_p).height = clamp_scalar(get_struct_value(Simulation_Params.persons(i_p), 'height', 1.70), 1.20, 2.30, 1.70);
        Simulation_Params.persons(i_p).weight = clamp_scalar(get_struct_value(Simulation_Params.persons(i_p), 'weight', 70.0), 30.0, 200.0, 70.0);
    end

    if ~isfield(Simulation_Params, 'objects') || isempty(Simulation_Params.objects)
        Simulation_Params.objects = [];
    else
        for i_o = 1:length(Simulation_Params.objects)
            if ~isfield(Simulation_Params.objects(i_o), 'name') || isempty(Simulation_Params.objects(i_o).name)
                Simulation_Params.objects(i_o).name = 'Unknown Object';
            end
            if ~isfield(Simulation_Params.objects(i_o), 'scatter_points') || isempty(Simulation_Params.objects(i_o).scatter_points)
                Simulation_Params.objects(i_o).scatter_points = [];
            else
                sp_matrix = double(Simulation_Params.objects(i_o).scatter_points);
                if size(sp_matrix, 2) < 3
                    sp_matrix = [];
                elseif size(sp_matrix, 2) == 3
                    sp_matrix = [sp_matrix, repmat(0.5, size(sp_matrix, 1), 1)];
                elseif size(sp_matrix, 2) > 4
                    sp_matrix = sp_matrix(:, 1:4);
                end
                if ~isempty(sp_matrix)
                    finite_rows = all(isfinite(sp_matrix), 2);
                    sp_matrix = sp_matrix(finite_rows, :);
                    if ~isempty(sp_matrix)
                        sp_matrix(:, 4) = min(max(sp_matrix(:, 4), 0.05), 5.0);
                    end
                end
                Simulation_Params.objects(i_o).scatter_points = sp_matrix;
            end
        end
    end
end

% Local helper routine safely reading a field from a structure with fallback defaults
function value = get_struct_value(input_struct, field_name, default_value)
    if isstruct(input_struct) && isfield(input_struct, field_name) && ~isempty(input_struct.(field_name))
        value = input_struct.(field_name);
    else
        value = default_value;
    end
end

% Local helper routine coercing arbitrary scalar data into bounded numeric values safely
function value = clamp_scalar(value_in, min_value, max_value, default_value)
    value = double(value_in);
    if isempty(value) || ~isscalar(value) || ~isfinite(value)
        value = default_value;
    end
    value = min(max(value, min_value), max_value);
end

% Local helper routine coercing flexible logical encodings into strict boolean flags
function bool_value = coerce_boolean(value_in)
    if islogical(value_in)
        bool_value = value_in;
    elseif isnumeric(value_in)
        bool_value = value_in ~= 0;
    elseif isstring(value_in) || ischar(value_in)
        str_value = lower(strtrim(string(value_in)));
        bool_value = any(strcmp(str_value, ["true", "1", "yes", "on"]));
    else
        bool_value = false;
    end
    bool_value = logical(bool_value);
end

% Local helper routine forcing vectors into finite row vectors of the requested dimension
function vec = ensure_row_vector(value_in, default_vec, required_length)
    vec = double(value_in(:)');
    if isempty(vec) || any(~isfinite(vec))
        vec = double(default_vec(:)');
    end
    if numel(vec) < required_length
        vec = [vec, default_vec(numel(vec)+1:required_length)];
    elseif numel(vec) > required_length
        vec = vec(1:required_length);
    end
end

% Local helper routine selecting the most recent newly generated or overwritten results file robustly
function latest_path = select_latest_results_file(pre_run_paths, pattern, run_start_time)
    latest_path = "";
    post_run_files = dir(pattern);
    if isempty(post_run_files)
        return;
    end

    post_run_paths = string(fullfile({post_run_files.folder}, {post_run_files.name}));
    new_indices = ~ismember(post_run_paths, pre_run_paths);
    candidate_files = post_run_files(new_indices);

    if isempty(candidate_files)
        tolerance_days = 5 / 86400;
        recent_indices = [post_run_files.datenum] >= (run_start_time - tolerance_days);
        candidate_files = post_run_files(recent_indices);
    end

    if isempty(candidate_files)
        return;
    end

    [~, idx_latest] = max([candidate_files.datenum]);
    latest_path = string(fullfile(candidate_files(idx_latest).folder, candidate_files(idx_latest).name));
end

% Local helper routine validating and homogenizing motion arrays into [Frames, Joints, XYZ] layout
function motion_mat = sanitize_motion_matrix(motion_in)
    motion_mat = squeeze(double(motion_in));
    if isempty(motion_mat) || any(~isfinite(motion_mat(:)))
        error("The loaded motion matrix is empty or contains invalid numeric values.");
    end

    if ndims(motion_mat) == 2
        if size(motion_mat, 2) == 66
            motion_mat = reshape(motion_mat, size(motion_mat, 1), 22, 3);
        elseif size(motion_mat, 1) == 66
            motion_mat = reshape(motion_mat', size(motion_mat, 2), 22, 3);
        else
            error("The loaded motion matrix is two dimensional but does not match an interpretable skeleton layout.");
        end
    end

    if ndims(motion_mat) ~= 3
        error("The loaded motion matrix does not have a valid three dimensional tensor layout.");
    end

    dims = size(motion_mat);
    coord_dim = find(dims == 3, 1, 'last');
    if isempty(coord_dim)
        error("The loaded motion matrix does not contain an identifiable XYZ coordinate dimension.");
    end
    if coord_dim ~= 3
        perm_order = 1:3;
        perm_order([coord_dim, 3]) = perm_order([3, coord_dim]);
        motion_mat = permute(motion_mat, perm_order);
        dims = size(motion_mat);
    end

    if dims(1) == 22 && dims(2) ~= 22
        motion_mat = permute(motion_mat, [2, 1, 3]);
        dims = size(motion_mat);
    elseif dims(2) ~= 22 && dims(1) ~= 22 && dims(2) < dims(1) && dims(2) <= 32
        motion_mat = permute(motion_mat, [2, 1, 3]);
        dims = size(motion_mat);
    end

    if dims(2) > 22
        motion_mat = motion_mat(:, 1:22, :);
    elseif dims(2) < 22
        pad_joint = repmat(motion_mat(:, end, :), [1, 22 - dims(2), 1]);
        motion_mat = cat(2, motion_mat, pad_joint);
    end
end

% Local helper routine deriving a stable odd Savitzky-Golay window length from the available frame count
function window_size = robust_sg_window(num_frames, requested_window)
    window_size = min(num_frames, requested_window);
    if mod(window_size, 2) == 0
        window_size = window_size - 1;
    end
    if window_size < 3
        window_size = 0;
    end
end

% Local helper routine estimating a person specific projected body radius from the kinematic envelope
function radius = estimate_person_footprint_radius(motion_mat)
    dx = motion_mat(:, :, 1) - motion_mat(:, 1, 1);
    dy = motion_mat(:, :, 3) - motion_mat(:, 1, 3);
    radial = sqrt(dx.^2 + dy.^2);
    radial = sort(radial(:));
    if isempty(radial)
        radius = 0.22;
        return;
    end
    idx = max(1, min(numel(radial), round(0.85 * numel(radial))));
    radius = radial(idx);
    radius = min(max(radius, 0.18), 0.45);
end

% Local helper routine projecting a circular person footprint outside an axis aligned bounding box
function [delta_x, delta_y, is_penetrating] = project_circle_out_of_bbox(curr_x, curr_y, clearance_radius, bbox)
    delta_x = 0;
    delta_y = 0;
    is_penetrating = false;

    x_min = bbox(1) - clearance_radius;
    x_max = bbox(2) + clearance_radius;
    y_min = bbox(3) - clearance_radius;
    y_max = bbox(4) + clearance_radius;

    if curr_x > x_min && curr_x < x_max && curr_y > y_min && curr_y < y_max
        dx1 = curr_x - x_min;
        dx2 = x_max - curr_x;
        dy1 = curr_y - y_min;
        dy2 = y_max - curr_y;
        [~, min_idx] = min([dx1, dx2, dy1, dy2]);

        if min_idx == 1
            delta_x = -dx1;
        elseif min_idx == 2
            delta_x = dx2;
        elseif min_idx == 3
            delta_y = -dy1;
        else
            delta_y = dy2;
        end
        is_penetrating = true;
    end
end

% Local helper routine estimating the scene center from all available physical entities
function scene_center = estimate_scene_center(Simulation_Params, tx_pos_list, rx_pos_list)
    point_cloud = [tx_pos_list; rx_pos_list];

    if isfield(Simulation_Params, 'persons') && ~isempty(Simulation_Params.persons)
        for i_p = 1:length(Simulation_Params.persons)
            if isfield(Simulation_Params.persons(i_p), 'start_pos') && ~isempty(Simulation_Params.persons(i_p).start_pos)
                point_cloud = [point_cloud; ensure_row_vector(Simulation_Params.persons(i_p).start_pos, [0, 0, 0], 3)]; %#ok<AGROW>
            end
            if isfield(Simulation_Params.persons(i_p), 'motion_mat') && ~isempty(Simulation_Params.persons(i_p).motion_mat)
                root_track = squeeze(Simulation_Params.persons(i_p).motion_mat(:, 1, :));
                if size(root_track, 2) == 3
                    root_track = root_track(:, [1, 3, 2]);
                end
                point_cloud = [point_cloud; root_track(1:max(1, floor(size(root_track, 1) / 10)):end, :)]; %#ok<AGROW>
            end
        end
    end

    if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
        for i_o = 1:length(Simulation_Params.objects)
            sp_matrix = Simulation_Params.objects(i_o).scatter_points;
            if ~isempty(sp_matrix) && size(sp_matrix, 2) >= 3
                point_cloud = [point_cloud; sp_matrix(:, 1:3)]; %#ok<AGROW>
            end
        end
    end

    if Simulation_Params.enable_wall
        point_cloud = [point_cloud; ensure_row_vector(Simulation_Params.wall_center, [0, 0, 1], 3)];
    end

    point_cloud = point_cloud(all(isfinite(point_cloud), 2), :);
    if isempty(point_cloud)
        scene_center = [0, 0, 1];
    else
        scene_center = mean(point_cloud, 1);
        if abs(scene_center(3)) < 1e-6
            scene_center(3) = 1.0;
        end
    end
end

% Local helper routine determining finite wall crossing distance, delay and amplitude loss for a direct path array securely mapping
function [d_wall, tau_wall_delay, trans_loss_amp, gamma_eff] = compute_wall_path_terms_vec(src_pos, dst_pos_mat, wall_center, wall_dimensions, v_wall, c, eps_r, wall_gamma)
    P = size(dst_pos_mat, 1);
    d_wall = zeros(P, 1);
    tau_wall_delay = zeros(P, 1);
    trans_loss_amp = ones(P, 1);
    gamma_eff = wall_gamma * ones(P, 1);
    
    delta_vec = dst_pos_mat - src_pos; % Px3
    norm_delta = sqrt(sum(delta_vec.^2, 2));
    
    valid_idx = norm_delta > 1e-9 & abs(delta_vec(:, 2)) > 1e-9;
    if ~any(valid_idx)
        return;
    end
    
    t_plane = zeros(P, 1);
    t_plane(valid_idx) = (wall_center(2) - src_pos(2)) ./ delta_vec(valid_idx, 2);
    
    cross_idx = valid_idx & (t_plane > 0) & (t_plane < 1);
    
    if ~any(cross_idx)
        return;
    end
    
    intersect_pt = src_pos + t_plane(cross_idx) .* delta_vec(cross_idx, :);
    x_min = wall_center(1) - wall_dimensions(1) / 2;
    x_max = wall_center(1) + wall_dimensions(1) / 2;
    z_min = wall_center(3) - wall_dimensions(3) / 2;
    z_max = wall_center(3) + wall_dimensions(3) / 2;
    
    hit_idx = intersect_pt(:, 1) >= x_min & intersect_pt(:, 1) <= x_max & ...
              intersect_pt(:, 3) >= z_min & intersect_pt(:, 3) <= z_max;
              
    final_cross_idx = find(cross_idx);
    final_hit_idx = final_cross_idx(hit_idx);
    
    if isempty(final_hit_idx)
        return;
    end
    
    cos_theta = abs(delta_vec(final_hit_idx, 2)) ./ norm_delta(final_hit_idx);
    cos_theta = max(cos_theta, 0.01);
    
    d_wall_hit = wall_dimensions(2) ./ cos_theta;
    d_wall(final_hit_idx) = d_wall_hit;
    tau_wall_delay(final_hit_idx) = d_wall_hit / v_wall - d_wall_hit / c;
    
    eps_r_val = max(double(eps_r), 1.0);
    sin_theta_sq = max(0.0, 1.0 - cos_theta.^2);
    root_term = sqrt(max(eps_r_val - sin_theta_sq, 0.0));
    
    gamma_te = (cos_theta - root_term) ./ (cos_theta + root_term + 1e-12);
    gamma_tm = (eps_r_val .* cos_theta - root_term) ./ (eps_r_val .* cos_theta + root_term + 1e-12);
    gamma_eff_hit = 0.5 * (gamma_te + gamma_tm);
    
    gamma_eff(final_hit_idx) = gamma_eff_hit;
    trans_loss_amp(final_hit_idx) = sqrt(max(1 - abs(gamma_eff_hit).^2, 0));
end

% Local helper routine determining finite wall crossing distance, delay and amplitude loss for a single static direct path
function [d_wall, tau_wall_delay, trans_loss_amp, gamma_eff, does_cross_wall] = compute_wall_path_terms(src_pos, dst_pos, wall_center, wall_dimensions, v_wall, c, eps_r)
    d_wall = 0.0;
    tau_wall_delay = 0.0;
    trans_loss_amp = 1.0;
    gamma_eff = 0.0;
    [does_cross_wall, cos_theta] = segment_crosses_wall(src_pos, dst_pos, wall_center, wall_dimensions);
    if does_cross_wall
        d_wall = wall_dimensions(2) / max(cos_theta, 0.01);
        tau_wall_delay = d_wall / v_wall - d_wall / c;
        gamma_eff = compute_dielectric_reflection_coeff(cos_theta, eps_r);
        trans_loss_amp = sqrt(max(1 - abs(gamma_eff)^2, 0));
    end
end

% Local helper routine evaluating whether a finite array path truly intersects the finite wall aperture securely replacing loops 
function is_valid = is_reflection_path_valid_vec(src_pos, image_target_pos_mat, wall_face_y, wall_center, wall_dimensions)
    P = size(image_target_pos_mat, 1);
    is_valid = false(P, 1);
    
    if isscalar(wall_face_y)
        wall_face_y = repmat(wall_face_y, P, 1);
    end
    
    delta_vec = image_target_pos_mat - src_pos; % Px3
    norm_delta = sqrt(sum(delta_vec.^2, 2));
    
    valid_idx = norm_delta > 1e-9 & abs(delta_vec(:, 2)) > 1e-9;
    if ~any(valid_idx)
        return;
    end
    
    t_plane = zeros(P, 1);
    t_plane(valid_idx) = (wall_face_y(valid_idx) - src_pos(2)) ./ delta_vec(valid_idx, 2);
    
    cross_idx = valid_idx & (t_plane > 0) & (t_plane < 1);
    
    if ~any(cross_idx)
        return;
    end
    
    intersect_pt = src_pos + t_plane(cross_idx) .* delta_vec(cross_idx, :);
    
    x_min = wall_center(1) - wall_dimensions(1) / 2;
    x_max = wall_center(1) + wall_dimensions(1) / 2;
    z_min = wall_center(3) - wall_dimensions(3) / 2;
    z_max = wall_center(3) + wall_dimensions(3) / 2;
    
    hit_idx = intersect_pt(:, 1) >= x_min & intersect_pt(:, 1) <= x_max & ...
              intersect_pt(:, 3) >= z_min & intersect_pt(:, 3) <= z_max;
              
    final_cross_idx = find(cross_idx);
    is_valid(final_cross_idx(hit_idx)) = true;
end

% Local helper routine evaluating whether a finite path truly intersects the finite wall aperture rather than an infinite wall plane
function [does_cross_wall, cos_theta] = segment_crosses_wall(src_pos, dst_pos, wall_center, wall_dimensions)
    does_cross_wall = false;
    cos_theta = 1.0;
    delta_vec = dst_pos - src_pos;
    if norm(delta_vec) < 1e-9 || abs(delta_vec(2)) < 1e-9
        return;
    end

    t_plane = (wall_center(2) - src_pos(2)) / delta_vec(2);
    if t_plane <= 0 || t_plane >= 1
        return;
    end

    intersect_pt = src_pos + t_plane * delta_vec;
    x_min = wall_center(1) - wall_dimensions(1) / 2;
    x_max = wall_center(1) + wall_dimensions(1) / 2;
    z_min = wall_center(3) - wall_dimensions(3) / 2;
    z_max = wall_center(3) + wall_dimensions(3) / 2;

    if intersect_pt(1) >= x_min && intersect_pt(1) <= x_max && intersect_pt(3) >= z_min && intersect_pt(3) <= z_max
        does_cross_wall = true;
        cos_theta = abs(delta_vec(2)) / norm(delta_vec);
    end
end

% Local helper routine evaluating whether a finite image theory reflection hits the actual static wall surface extent
function is_valid = is_reflection_path_valid(src_pos, image_target_pos, wall_face_y, wall_center, wall_dimensions)
    is_valid = false;
    delta_vec = image_target_pos - src_pos;
    if norm(delta_vec) < 1e-9 || abs(delta_vec(2)) < 1e-9
        return;
    end

    t_plane = (wall_face_y - src_pos(2)) / delta_vec(2);
    if t_plane <= 0 || t_plane >= 1
        return;
    end

    intersect_pt = src_pos + t_plane * delta_vec;
    x_min = wall_center(1) - wall_dimensions(1) / 2;
    x_max = wall_center(1) + wall_dimensions(1) / 2;
    z_min = wall_center(3) - wall_dimensions(3) / 2;
    z_max = wall_center(3) + wall_dimensions(3) / 2;

    is_valid = intersect_pt(1) >= x_min && intersect_pt(1) <= x_max && intersect_pt(3) >= z_min && intersect_pt(3) <= z_max;
end

% Local helper routine approximating an incidence angle dependent dielectric reflection coefficient
function gamma_eff = compute_dielectric_reflection_coeff(cos_theta, eps_r)
    cos_theta = min(max(cos_theta, 0.0), 1.0);
    eps_r = max(double(eps_r), 1.0);
    sin_theta_sq = max(0.0, 1.0 - cos_theta^2);
    root_term = sqrt(max(eps_r - sin_theta_sq, 0.0));

    gamma_te = (cos_theta - root_term) / (cos_theta + root_term + 1e-12);
    gamma_tm = (eps_r * cos_theta - root_term) / (eps_r * cos_theta + root_term + 1e-12);
    gamma_eff = 0.5 * (gamma_te + gamma_tm);
end

% Local helper routine safely normalizing logarithmic matrices for RGB export avoiding NaNs and out of range pixels
function normalized_img = normalize_log_image_for_export(Matrix_Log, log_clim_min, log_clim_max)
    normalized_img = (double(Matrix_Log) - log_clim_min) / max(log_clim_max - log_clim_min, 1e-12);
    normalized_img(~isfinite(normalized_img)) = 0;
    normalized_img = min(max(normalized_img, 0), 1);
end