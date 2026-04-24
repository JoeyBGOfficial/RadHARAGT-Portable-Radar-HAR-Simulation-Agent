function RadHARAGT_GUI()
%% RadHARAGT Graphical User Interface Main Application
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2026-04-24.
% Affiliate: Beijing Institute of Technology.
% Platform: Ollama, MATLAB R2025b, Python 3.10.13 with Conda EMDM Environment.
%
% Introduction:
%   This interactive graphical application extends the comprehensive pipeline for generating realistic indoor multi-human radar echo datasets.
%   It features a modern conversational interface bridging continuous contextual multimodal descriptions to a local large language model.
%   The script interprets multi-turn natural language to extract precise radar simulation parameters tracking unified temporal physical bounds and static structural objects continuously.
%   It utilizes a python based motion diffusion model to generate three dimensional trajectories recursively evaluated through spatial temporal collision enforcement models.
%   The system simulates multi channel frequency modulated continuous wave radar raw traces precisely predicting dynamic micro Doppler signatures.
%   It calculates robust spatial spectral matrices mapping First Channel and Channel Sum features systematically isolating background statics securely.
%   The application concludes by integrating isolated spatial spectrogram plots and animated physical rendering sequences natively within the conversational dialogue history.

    %% Initialization and Global State Definition
    close all; 
    clc;

    % Establish universal graphical rendering definitions securing unified layout aesthetics across components
    Font_Name = 'Palatino Linotype';
    Font_Name_Terminal = 'Palatino Linotype';
    Font_Size_Basis = 14;
    Font_Size_Axis = 15;
    Font_Size_Title = 16;
    Font_Weight_Basis = 'normal';
    Font_Weight_Axis = 'normal';
    Font_Weight_Title = 'bold';

    % Allocate global palette specifications mapping exact visual hierarchies structurally
    Color_Background = [1.00 1.00 1.00];
    Color_Border = [0.88 0.88 0.88];
    Color_Panel_Think = [0.97 0.98 1.00];
    Color_Border_Think = [0.80 0.85 0.95];
    Color_Panel_Result = [0.95 0.98 0.96];
    Color_Border_Result = [0.85 0.92 0.86];
    Color_Control_Soft = [0.97 0.98 1.00];
    Color_Control_Button = [0.95 0.96 0.98];
    Color_Control_Accent = [0.15 0.40 0.70];
    Color_Control_Warm = [1.00 0.93 0.93];
    Color_Control_Warm_Text = [0.78 0.22 0.25];
    Color_Control_Processing = [0.98 0.92 0.84];
    Color_Control_Processing_Text = [0.75 0.45 0.10];
    Color_Output_Button = [0.93 0.97 0.93];
    Color_Control_Label = [0.30 0.30 0.30];
    Color_Control_Muted = [0.40 0.45 0.52];
    Color_Control_Disabled_Label = [0.72 0.74 0.78];
    Color_Control_Disabled_Field = [0.96 0.96 0.97];
    Color_Control_Disabled_Text = [0.69 0.71 0.75];

    % Initialize the application multimodal tracking structures holding contextual records securely
    Chat_History = struct('role', {}, 'content', {}, 'images', {});
    Current_Image_Path = "";
    Has_Image = false;
    Default_Conda_Env = "emdm";
    Latest_Save_Dir = "";
    Latest_State_JSON = "";
    Latest_State_Params = struct();
    Has_Latest_State = false;

    % Store proprietary colormap arrays resolving radar signature mappings universally
    JoeyBG_Colormap = [0.6196 0.0039 0.2588; 0.8353 0.2431 0.3098; 0.9569 0.4275 0.2627; ...
                       0.9922 0.6824 0.3804; 0.9961 0.8784 0.5451; 1.0000 1.0000 0.7490; ...
                       0.9020 0.9608 0.5961; 0.6706 0.8667 0.6431; 0.4000 0.7608 0.6471; ...
                       0.1961 0.5333 0.7412; 0.3686 0.3098 0.6353];
    JoeyBG_Colormap_Flip = flip(JoeyBG_Colormap);

    %% Construct Foundation Intelligence Rules
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
        '  "antenna_beamwidth": 60,\n', ...
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

    System_Prompt = strcat("你现在是一个专业的Human Motion生成与雷达仿真物理参数提取专家。\n"...
        , "你的核心任务是理解用户对室内人体动作与雷达探测场景的自然语言描述并将其精准地转化为结构化的JSON参数字典。\n\n"...
        , "任务一 多目标动作参数提取 persons数组\n"...
        , "如果场景中包含多个人物请将他们分离开并存入persons列表数组中哪怕只有一个人物也要放入列表中。每个person需要精准提取以下参数\n"...
        , "- Refined_Prompt 将该人物的基础动作描述翻译并润色为HumanML3D描述风格的英文Prompt长度为十到三十词。对于跌倒或者坐下等有明确终态的动作必须明确指出速度和最终姿态。\n"...
        , "- start_pos 动作起始三维绝对坐标XYZ单位为米。如果没有明确指定默认值为原点。\n"...
        , "- start_heading 起始运动朝向的偏航角单位为度零代表沿X轴正向九十代表沿Y轴正向默认为零。\n"...
        , "- start_time_delay 动作的起始时间间隔或延迟单位为秒若未提及默认为零。\n"...
        , "- height 身高单位为米用于计算雷达截面积若未明确默认一点七零。\n"...
        , "- weight 体重单位为公斤用于计算雷达截面积若未明确默认七十。\n\n"...
        , "任务二 雷达与物理场景参数提取\n"...
        , "从描述中提取雷达射频参数、天线位置及场景媒质属性。请严格遵循以下物理单位和坐标系定义若用户未提供某些参数请补充合理默认值\n"...
        , "- 射频参数必须转换为基本单位 中心频率fc单位为赫兹例如两吉赫兹必须输出为2000000000。带宽B单位为赫兹。脉冲宽度tp单位为秒默认千分之一。脉冲重复频率PRF单位为赫兹。采样率fs单位为赫兹默认一千万。\n"...
        , "- 系统与天线 系统信噪比SNR单位为分贝默认五十。天线增益antenna_gain单位为dBi默认十五。天线波束宽度antenna_beamwidth单位为度默认六十。收发隔离度antenna_isolation单位为分贝默认四十。\n"...
        , "- 坐标系定义绝对准则 所有的三维向量必须严格遵守X代表方位宽度、Y代表距离深度厚度、Z代表高度的笛卡尔物理坐标系。\n"...
        , "- 天线位置 发射天线tx_pos和接收天线rx_pos格式必须为包含XYZ的二维数组单位为米。哪怕是单天线也必须嵌套一层列表多发多收则输出多个列表以支持MIMO阵列结构提取。\n"...
        , "- 墙体参数 若语义中包含墙体则enable_wall等于true墙体中心wall_center为包含三个坐标分量的数组。墙体尺寸wall_dimensions必须且只能按照X轴宽度、Y轴厚度、Z轴高度的顺序输出。例如用户说明宽五米高三米厚零点二米必须输出为包含五点零、零点二和三点零的数组。墙体介电常数wall_epsilon_r和损耗角正切wall_loss_tangent按原意提取。\n"...
        , "- 信号处理 stft_window_seconds单位为秒默认零点一，stft_overlap_ratio介于零和一之间默认零点七五。\n"...
        , "- 功能开关使用布尔值true或false 是否启用墙体enable_wall、是否考虑多径效应enable_multipath、是否使用神经网络进行特征增强去噪enable_network。如果用户明确说不使用某项功能则设为false。\n\n"...
        , "任务三 静止简单物体与关键散射点提取 objects数组\n"...
        , "如果用户描述了房间内另外放置了简单常见物体例如桌子、椅子、沙发或柜子等请将它们存入objects列表数组中若无此类物体则返回空列表。每个object需要提取并推演生成\n"...
        , "- name 物体名称的英文描述类似于Wooden Desk。\n"...
        , "- scatter_points 必须是一个二维数组每一行代表大模型为你推演出的该物体上的一个关键雷达散射点格式严格为XYZ和RCS。其中XYZ为该点在场景中的绝对三维笛卡尔物理坐标单位为米，RCS为该散射点对应的雷达截面积估值单位为平方米典型值分布于零点一至二点零之间。\n"...
        , "- 请充分根据描述中物体的物理长宽高尺寸和指定方位利用你的三维空间几何理解能力自动为其生成分布合理的关键离散散射点例如一张长一点二米的桌子应包含四个桌角坐标和桌面中心坐标等从而利用离散点云完美近似勾勒出该物体的三维空间轮廓和电磁反射特征这对于雷达仿真至关重要。\n\n"...
        , "输出格式严格约束\n"...
        , "你的输出必须且只能是一个合法的JSON对象完全匹配下方模板的键值名称和数据类型。绝对不要输出任何额外的思考过程、问候语或Markdown标记。\n"...
        , "模板示例\n", JSON_Template);
        
    Chat_History(1).role = 'system';
    Chat_History(1).content = System_Prompt;
    Chat_History(1).images = cell(1,0);

    %% Construct Primary User Interface Window
    main_fig = uifigure('Name', 'RadHARAGT', 'Color', Color_Background, 'Position', [100, 50, 1500, 950]);

    % Launch the primary window in maximized desktop state while preserving normal window controls for later manual resizing
    main_fig.WindowState = 'maximized';

    % Apply the user supplied desktop logo securing unified window branding across local sessions
    if isfile('RadHARAGT_Logo.png')
        main_fig.Icon = 'RadHARAGT_Logo.png';
    end

    main_grid = uigridlayout(main_fig, [3, 1], 'RowHeight', {110, '1x', 140}, 'ColumnWidth', {'1x'}, 'BackgroundColor', Color_Background);

    % Build Header Section securing symmetric title positioning safely resolving dimensions
    header_grid = uigridlayout(main_grid, [2, 1], 'RowHeight', {'1x', 25}, 'ColumnWidth', {'1x'}, 'BackgroundColor', Color_Background);
    header_grid.Layout.Row = 1; 
    header_grid.Layout.Column = 1;
    
    if isfile('RadHARAGT_Interface.png')
        img_title = uiimage(header_grid, 'ImageSource', 'RadHARAGT_Interface.png');
        img_title.Layout.Row = 1;
        img_title.Layout.Column = 1;
    end
    
    lbl_author = uilabel(header_grid, 'Text', 'Author: JoeyBG, Affiliation: Beijing Institute of Technology', 'FontName', Font_Name, 'FontSize', 13, 'FontColor', [0.4 0.4 0.4], 'HorizontalAlignment', 'center');
    lbl_author.Layout.Row = 2; 
    lbl_author.Layout.Column = 1;

    % Build Scrollable Conversational Area directly utilizing native scrollable grid layouts bypassing clipping limits completely
    chat_layout = uigridlayout(main_grid, [1, 1], 'RowHeight', {'fit'}, 'ColumnWidth', {'1x'}, 'BackgroundColor', Color_Background, 'RowSpacing', 15, 'Scrollable', 'on');
    chat_layout.Layout.Row = 2; 
    chat_layout.Layout.Column = 1;

    % Build Interactive Input Control Area enforcing compact alignment across all bottom command modules
    input_panel = uipanel(main_grid, 'BackgroundColor', Color_Background, 'BorderType', 'line', 'BorderColor', Color_Border);
    input_panel.Layout.Row = 3; 
    input_panel.Layout.Column = 1;
    
    input_grid = uigridlayout(input_panel, [2, 1], 'RowHeight', {'1x', 45}, 'ColumnWidth', {'1x'}, 'BackgroundColor', Color_Background, 'Padding', [10 10 10 10], 'RowSpacing', 8);
    
    prompt_area = uitextarea(input_grid, 'FontName', Font_Name, 'FontSize', 14, 'Placeholder', 'Enter your natural language scenario description here and trigger the simulation generator...');
    prompt_area.Layout.Row = 1; 
    prompt_area.Layout.Column = 1;

    % Arrange the lower command strip with balanced spacing preserving clean horizontal rhythm across five modules
    ctrl_grid = uigridlayout(input_grid, [1, 5], 'RowHeight', {'1x'}, 'ColumnWidth', {'0.85x', '1.00x', '1.02x', '1.95x', '0.85x'}, 'Padding', [0 0 0 0], 'ColumnSpacing', 8, 'BackgroundColor', Color_Background);
    ctrl_grid.Layout.Row = 2;
    ctrl_grid.Layout.Column = 1;

    btn_img = uibutton(ctrl_grid, 'push', 'Text', '  Select Image', 'FontName', Font_Name, 'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0.15 0.15 0.15], 'BackgroundColor', Color_Control_Button, 'ButtonPushedFcn', @cb_select_image);
    if isfile('Image.png'), btn_img.Icon = 'Image.png'; btn_img.IconAlignment = 'left'; end
    btn_img.Layout.Row = 1; 
    btn_img.Layout.Column = 1;
    btn_img.Tooltip = 'Select a reference image from the current local drive.';

    model_grid = uigridlayout(ctrl_grid, [1, 2], 'RowHeight', {'1x'}, 'ColumnWidth', {54, '1x'}, 'Padding', [0 0 0 0], 'ColumnSpacing', 6, 'BackgroundColor', Color_Background);
    model_grid.Layout.Row = 1; 
    model_grid.Layout.Column = 2;
    
    lbl_model = uilabel(model_grid, 'Text', 'Model:', 'FontName', Font_Name, 'FontSize', 14, 'FontColor', Color_Control_Label, 'HorizontalAlignment', 'right');
    lbl_model.Layout.Row = 1; 
    lbl_model.Layout.Column = 1;
    
    edit_model = uieditfield(model_grid, 'text', 'Value', 'gemma4:e4b', 'FontName', Font_Name, 'FontSize', 14, 'BackgroundColor', [1 1 1]);
    edit_model.Layout.Row = 1; 
    edit_model.Layout.Column = 2;

    conda_grid = uigridlayout(ctrl_grid, [1, 2], 'RowHeight', {'1x'}, 'ColumnWidth', {82, '1x'}, 'Padding', [0 0 0 0], 'ColumnSpacing', 6, 'BackgroundColor', Color_Background);
    conda_grid.Layout.Row = 1; 
    conda_grid.Layout.Column = 3;

    lbl_conda = uilabel(conda_grid, 'Text', 'Conda Env.:', 'FontName', Font_Name, 'FontSize', 14, 'FontColor', Color_Control_Label, 'HorizontalAlignment', 'right');
    lbl_conda.Layout.Row = 1; 
    lbl_conda.Layout.Column = 1;

    edit_conda = uieditfield(conda_grid, 'text', 'Value', char(Default_Conda_Env), 'FontName', Font_Name, 'FontSize', 14, 'BackgroundColor', [1 1 1]);
    edit_conda.Layout.Row = 1; 
    edit_conda.Layout.Column = 2;

    temp_grid = uigridlayout(ctrl_grid, [1, 5], 'RowHeight', {'1x'}, 'ColumnWidth', {86, 52, 18, '1x', 24}, 'Padding', [0 0 0 0], 'ColumnSpacing', 5, 'BackgroundColor', Color_Background);
    temp_grid.Layout.Row = 1; 
    temp_grid.Layout.Column = 4;
    
    lbl_temp_title = uilabel(temp_grid, 'Text', 'Temperature:', 'FontName', Font_Name, 'FontSize', 14, 'FontColor', Color_Control_Label, 'HorizontalAlignment', 'right');
    lbl_temp_title.Layout.Row = 1; 
    lbl_temp_title.Layout.Column = 1;

    edit_temp_val = uieditfield(temp_grid, 'text', 'Value', '5.0', 'Editable', 'off', 'FontName', Font_Name, 'FontSize', 14, 'FontWeight', 'bold', 'FontColor', Color_Control_Accent, 'BackgroundColor', Color_Control_Soft, 'HorizontalAlignment', 'center');
    edit_temp_val.Layout.Row = 1; 
    edit_temp_val.Layout.Column = 2;
    
    lbl_temp_min = uilabel(temp_grid, 'Text', '1', 'FontName', Font_Name, 'FontSize', 12, 'FontColor', Color_Control_Muted, 'HorizontalAlignment', 'center');
    lbl_temp_min.Layout.Row = 1; 
    lbl_temp_min.Layout.Column = 3;
    
    slider_temp = uislider(temp_grid, 'Limits', [1, 10], 'Value', 5.0, 'MajorTicks', [], 'MinorTicks', []);
    slider_temp.ValueChangingFcn = @(~, event) update_temp_lbl(edit_temp_val, event.Value);
    slider_temp.ValueChangedFcn = @(~, event) update_temp_lbl(edit_temp_val, event.Value);
    slider_temp.Layout.Row = 1; 
    slider_temp.Layout.Column = 4;
    
    lbl_temp_max = uilabel(temp_grid, 'Text', '10', 'FontName', Font_Name, 'FontSize', 12, 'FontColor', Color_Control_Muted, 'HorizontalAlignment', 'center');
    lbl_temp_max.Layout.Row = 1; 
    lbl_temp_max.Layout.Column = 5;

    btn_gen = uibutton(ctrl_grid, 'push', 'Text', 'Generate  ', 'FontName', Font_Name, 'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0.15 0.15 0.15], 'BackgroundColor', Color_Control_Button, 'ButtonPushedFcn', @cb_generate);
    if isfile('Generate.png'), btn_gen.Icon = 'Generate.png'; btn_gen.IconAlignment = 'right'; end
    btn_gen.Layout.Row = 1; 
    btn_gen.Layout.Column = 5;

    %% Graphical Callback Functions
    function scroll_chat_view_safely()
        drawnow limitrate;
        pause(0.02);
        scroll(chat_layout, 'bottom');
        drawnow;
    end

    function update_temp_lbl(value_obj, new_val)
        value_obj.Value = sprintf('%.1f', new_val);
    end

    function set_image_button_idle_state()
        btn_img.Text = '  Select Image';
        btn_img.FontColor = [0.15 0.15 0.15];
        btn_img.BackgroundColor = Color_Control_Button;
        btn_img.Tooltip = 'Select a reference image from the current local drive.';
    end

    function set_image_button_loaded_state()
        btn_img.Text = '  Image Loaded';
        btn_img.FontColor = Color_Control_Warm_Text;
        btn_img.BackgroundColor = Color_Control_Warm;
        btn_img.Tooltip = 'Click once more to remove the loaded reference image.';
    end

    % Synchronize the complete bottom command strip visual state preserving consistent muted rendering while the solver is busy
    function set_command_strip_enabled(is_enabled)
        if is_enabled
            main_fig.Pointer = 'arrow';
            btn_gen.Enable = 'on';
            btn_gen.Text = 'Generate  ';
            if isfile('Generate.png'), btn_gen.Icon = 'Generate.png'; end
            btn_gen.BackgroundColor = Color_Control_Button;
            btn_gen.FontColor = [0.15 0.15 0.15];
            
            btn_img.Enable = 'on';
            prompt_area.Enable = 'on';
            edit_model.Enable = 'on';
            edit_conda.Enable = 'on';
            slider_temp.Enable = 'on';
            edit_temp_val.Enable = 'on';

            lbl_model.FontColor = Color_Control_Label;
            lbl_conda.FontColor = Color_Control_Label;
            lbl_temp_title.FontColor = Color_Control_Label;
            lbl_temp_min.FontColor = Color_Control_Muted;
            lbl_temp_max.FontColor = Color_Control_Muted;

            edit_model.BackgroundColor = [1 1 1];
            edit_model.FontColor = [0.15 0.15 0.15];
            edit_conda.BackgroundColor = [1 1 1];
            edit_conda.FontColor = [0.15 0.15 0.15];
            edit_temp_val.BackgroundColor = Color_Control_Soft;
            edit_temp_val.FontColor = Color_Control_Accent;
        else
            main_fig.Pointer = 'watch';
            btn_gen.Enable = 'off';
            btn_gen.Text = 'Processing...';
            btn_gen.Icon = '';
            btn_gen.BackgroundColor = Color_Control_Processing;
            btn_gen.FontColor = Color_Control_Processing_Text;
            
            btn_img.Enable = 'off';
            prompt_area.Enable = 'off';
            edit_model.Enable = 'off';
            edit_conda.Enable = 'off';
            slider_temp.Enable = 'off';
            edit_temp_val.Enable = 'off';

            lbl_model.FontColor = Color_Control_Disabled_Label;
            lbl_conda.FontColor = Color_Control_Disabled_Label;
            lbl_temp_title.FontColor = Color_Control_Disabled_Label;
            lbl_temp_min.FontColor = Color_Control_Disabled_Label;
            lbl_temp_max.FontColor = Color_Control_Disabled_Label;

            edit_model.BackgroundColor = Color_Control_Disabled_Field;
            edit_model.FontColor = Color_Control_Disabled_Text;
            edit_conda.BackgroundColor = Color_Control_Disabled_Field;
            edit_conda.FontColor = Color_Control_Disabled_Text;
            edit_temp_val.BackgroundColor = Color_Control_Disabled_Field;
            edit_temp_val.FontColor = Color_Control_Disabled_Text;
        end
    end

    function cb_select_image(~, ~)
        if Has_Image && strlength(Current_Image_Path) > 0
            Has_Image = false;
            Current_Image_Path = "";
            set_image_button_idle_state();
            return;
        end

        [file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'});
        if ~isequal(file, 0)
            Current_Image_Path = fullfile(path, file);
            Has_Image = true;
            set_image_button_loaded_state();
        end
    end

    % Open the most recent simulation result folder directly from the output panel for immediate inspection and export handling
    function cb_open_latest_folder(~, ~)
        if strlength(Latest_Save_Dir) == 0 || ~isfolder(Latest_Save_Dir)
            return;
        end

        folder_path = char(Latest_Save_Dir);
        if ispc
            winopen(folder_path);
        elseif ismac
            system(sprintf('open "%s" &', strrep(folder_path, '"', '\"')));
        else
            system(sprintf('xdg-open "%s" >/dev/null 2>&1 &', strrep(folder_path, '"', '\"')));
        end
    end

    function cb_generate(~, ~)
        input_text = prompt_area.Value;
        if iscell(input_text)
            input_text = strjoin(input_text, char(10));
        end
        input_text = strtrim(input_text);
        if isempty(input_text)
            return;
        end
        
        set_command_strip_enabled(false);
        
        % Secure the main window interface pointer communicating a busy processing state clearly preventing duplicate accidental triggers
        ptr_cleanup = onCleanup(@() set(main_fig, 'Pointer', 'arrow'));
        
        Model_Name = edit_model.Value;
        Conda_Env_Name = strtrim(string(edit_conda.Value));
        if strlength(Conda_Env_Name) == 0
            Conda_Env_Name = Default_Conda_Env;
            edit_conda.Value = char(Conda_Env_Name);
        end
        Guidance_Param = num2str(slider_temp.Value, '%.1f');
        
        append_you_bubble(input_text, Has_Image, Current_Image_Path);
        think_area = append_think_bubble();
        
        user_msg = struct('role', 'user', 'content', input_text, 'images', {cell(1,0)});
        if Has_Image && isfile(Current_Image_Path)
            fid = fopen(Current_Image_Path, 'rb');
            imgData = fread(fid, '*uint8');
            fclose(fid);
            user_msg.images = {matlab.net.base64encode(imgData)};
        end
        Chat_History(end+1) = user_msg;
        
        prompt_area.Value = '';
        Has_Image = false;
        Current_Image_Path = "";
        set_image_button_idle_state();
        drawnow limitrate;
        
        try
            execute_simulation(think_area, Model_Name, Guidance_Param, Conda_Env_Name);
        catch ME
            agent_think_stream(think_area, "A severe procedural violation occurred abruptly halting the algorithmic flow. Printing system trace below for structural debugging.");
            agent_think_stream(think_area, string(ME.message));
            uialert(main_fig, ME.message, 'Simulation Interrupted', 'Icon', 'error');
        end
        
        set_command_strip_enabled(true);
        
        drawnow limitrate; pause(0.05); scroll(chat_layout, 'bottom');
    end

    %% Internal Component Builders Safely Isolating Property Maps
    function append_you_bubble(text_str, has_img, img_path)
        estimated_text_height = max(50, ceil(strlength(text_str) / 80) * 20); 
        bubble_height = 60 + estimated_text_height;
        if has_img && isfile(img_path)
            bubble_height = bubble_height + 230;
        end
        
        curr_rows = chat_layout.RowHeight;
        curr_rows{end+1} = bubble_height;
        chat_layout.RowHeight = curr_rows;
        
        pnl = uipanel(chat_layout, 'BackgroundColor', Color_Background, 'BorderType', 'line', 'BorderColor', Color_Border);
        pnl.Layout.Row = length(chat_layout.RowHeight);
        pnl.Layout.Column = 1;
        
        g = uigridlayout(pnl, [2, 2], 'RowHeight', {30, '1x'}, 'ColumnWidth', {35, '1x'}, 'BackgroundColor', Color_Background);
        
        if isfile('You.png')
            img_you = uiimage(g, 'ImageSource', 'You.png');
            img_you.Layout.Row = 1;
            img_you.Layout.Column = 1;
        end
        
        lbl_you = uilabel(g, 'Text', 'You', 'FontName', Font_Name, 'FontWeight', 'bold', 'FontSize', 15);
        lbl_you.Layout.Row = 1;
        lbl_you.Layout.Column = 2;
        
        content_g = uigridlayout(g, [2, 1], 'RowHeight', {'1x', 'fit'}, 'Padding', [0 0 0 0], 'BackgroundColor', Color_Background);
        content_g.Layout.Row = 2; 
        content_g.Layout.Column = [1, 2];
        
        txt_content = uitextarea(content_g, 'Value', text_str, 'FontName', Font_Name, 'FontSize', 14, 'Editable', 'off', 'BackgroundColor', Color_Background);
        txt_content.Layout.Row = 1;
        txt_content.Layout.Column = 1;
        
        if has_img && isfile(img_path)
            content_g.RowHeight{2} = 220;
            img_ref = uiimage(content_g, 'ImageSource', img_path, 'HorizontalAlignment', 'left');
            img_ref.Layout.Row = 2; 
            img_ref.Layout.Column = 1;
        end
        drawnow limitrate; pause(0.05); scroll(chat_layout, 'bottom');
    end

    function think_area = append_think_bubble()
        curr_rows = chat_layout.RowHeight;
        curr_rows{end+1} = 280;
        chat_layout.RowHeight = curr_rows;
        
        pnl = uipanel(chat_layout, 'BackgroundColor', Color_Panel_Think, 'BorderType', 'line', 'BorderColor', Color_Border_Think);
        pnl.Layout.Row = length(chat_layout.RowHeight);
        pnl.Layout.Column = 1;
        
        g = uigridlayout(pnl, [2, 2], 'RowHeight', {30, '1x'}, 'ColumnWidth', {35, '1x'}, 'BackgroundColor', Color_Panel_Think);
        
        if isfile('Think.png')
            img_think = uiimage(g, 'ImageSource', 'Think.png');
            img_think.Layout.Row = 1;
            img_think.Layout.Column = 1;
        end
        
        lbl_think = uilabel(g, 'Text', 'Agent Working', 'FontName', Font_Name, 'FontWeight', 'bold', 'FontSize', 15, 'FontColor', Color_Control_Accent);
        lbl_think.Layout.Row = 1; 
        lbl_think.Layout.Column = 2;
        
        think_area = uitextarea(g, 'Value', {''}, 'FontName', Font_Name_Terminal, 'FontSize', 13, 'Editable', 'off', 'BackgroundColor', Color_Panel_Think, 'FontColor', [0.2 0.2 0.2]);
        think_area.Layout.Row = 2; 
        think_area.Layout.Column = [1, 2];
        
        drawnow limitrate; pause(0.05); scroll(chat_layout, 'bottom');
    end

    function [ax_rtm1, ax_dtm1, ax_rtms, ax_dtms, ax_3d] = append_result_bubble()
        curr_rows = chat_layout.RowHeight;
        curr_rows{end+1} = 450; 
        chat_layout.RowHeight = curr_rows;
        
        pnl = uipanel(chat_layout, 'BackgroundColor', Color_Panel_Result, 'BorderType', 'line', 'BorderColor', Color_Border_Result);
        pnl.Layout.Row = length(chat_layout.RowHeight);
        pnl.Layout.Column = 1;
        
        g = uigridlayout(pnl, [3, 3], 'RowHeight', {30, 25, '1x'}, 'ColumnWidth', {35, '1x', 130}, 'BackgroundColor', Color_Panel_Result, 'ColumnSpacing', 8);
        
        if isfile('Agent.png')
            img_agent = uiimage(g, 'ImageSource', 'Agent.png');
            img_agent.Layout.Row = 1;
            img_agent.Layout.Column = 1;
        end
        
        lbl_res_title = uilabel(g, 'Text', 'Agent Output', 'FontName', Font_Name, 'FontWeight', 'bold', 'FontSize', 15, 'FontColor', [0.15 0.60 0.20]);
        lbl_res_title.Layout.Row = 1; 
        lbl_res_title.Layout.Column = 2;

        btn_open_folder = uibutton(g, 'push', 'Text', 'Open Folder', 'FontName', Font_Name, 'FontSize', 13, 'FontColor', [0.12 0.28 0.12], 'BackgroundColor', Color_Output_Button, 'ButtonPushedFcn', @cb_open_latest_folder);
        btn_open_folder.Layout.Row = 1;
        btn_open_folder.Layout.Column = 3;
        if strlength(Latest_Save_Dir) == 0 || ~isfolder(Latest_Save_Dir)
            btn_open_folder.Enable = 'off';
        end
        
        lbl_res_desc = uilabel(g, 'Text', 'Simulation completed successfully. Generated radar signatures and motion visualization for the scenario.', 'FontName', Font_Name, 'FontSize', 13, 'FontColor', [0.35 0.45 0.35]);
        lbl_res_desc.Layout.Row = 2; 
        lbl_res_desc.Layout.Column = [1, 3];
        
        res_grid = uigridlayout(g, [1, 5], 'RowHeight', {'1x'}, 'ColumnWidth', {'1x', '1x', '1x', '1x', '1.6x'}, 'Padding', [0 0 0 0], 'BackgroundColor', Color_Panel_Result);
        res_grid.Layout.Row = 3; 
        res_grid.Layout.Column = [1, 3];
        
        ax_rtm1 = uiaxes(res_grid, 'Color', Color_Panel_Result); ax_rtm1.Layout.Column = 1; ax_rtm1.Layout.Row = 1;
        ax_dtm1 = uiaxes(res_grid, 'Color', Color_Panel_Result); ax_dtm1.Layout.Column = 2; ax_dtm1.Layout.Row = 1;
        ax_rtms = uiaxes(res_grid, 'Color', Color_Panel_Result); ax_rtms.Layout.Column = 3; ax_rtms.Layout.Row = 1;
        ax_dtms = uiaxes(res_grid, 'Color', Color_Panel_Result); ax_dtms.Layout.Column = 4; ax_dtms.Layout.Row = 1;
        ax_3d = uiaxes(res_grid, 'Color', Color_Panel_Result); ax_3d.Layout.Column = 5; ax_3d.Layout.Row = 1;
        
        drawnow limitrate; pause(0.05); scroll(chat_layout, 'bottom');
    end

    % Smooth and asynchronous character rendering sequence effectively avoiding native interface thread locking
    function agent_think_stream(textarea_obj, text_str)
        time_str = char(datetime('now', 'Format', 'HH:mm:ss'));
        full_line = sprintf('[%s] %s', time_str, char(text_str));
        
        current_val = textarea_obj.Value;
        if isempty(current_val)
            current_val = {''};
        elseif ischar(current_val) || isstring(current_val)
            current_val = cellstr(current_val);
        end

        is_initial_blank = numel(current_val) == 1 && strlength(string(current_val{1})) == 0;
        if is_initial_blank
            current_val{1} = '';
        else
            current_val{end+1} = '';
        end
        
        char_array = char(full_line);
        update_timer = tic;
        
        for idx = 1:length(char_array)
            current_val{end} = char_array(1:idx);
            if toc(update_timer) > 0.03 || idx == length(char_array)
                textarea_obj.Value = current_val;
                scroll(textarea_obj, 'bottom');
                drawnow limitrate;
                update_timer = tic;
            end
        end
    end

    %% Core Deep Mathematical Radar Simulation Engine
    function execute_simulation(think_area, Model_Name, Guidance_Param, Conda_Env)
        Random_Seed = 42;
        rng(Random_Seed, 'twister');
        
        EMDM_Path = "EMDM";
        Conda_Env = string(strtrim(char(string(Conda_Env))));
        if strlength(Conda_Env) == 0
            Conda_Env = Default_Conda_Env;
        end
        Motion_FPS = 20;
        Current_User_Text = string(Chat_History(end).content);

        current_time_str = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        Save_Dir = sprintf('Simulation_Result_%s', current_time_str);
        if ~exist(Save_Dir, 'dir')
            mkdir(Save_Dir);
        end
        Latest_Save_Dir = string(Save_Dir);
        agent_think_stream(think_area, sprintf("Allocating local workspace directory at %s to securely store all generated simulation outputs.", Save_Dir));
        
        agent_think_stream(think_area, "Initiating connection to the local large language model to comprehend the scene description and extract required simulation parameters.");
        
        apiEndpoint = "http://localhost:11434/api/chat";
        options = weboptions('MediaType', 'application/json', 'Timeout', 180);
        
        agent_think_stream(think_area, "Transmitting the multimodal request payload to the local language model and awaiting cognitive response.");
        try
            [LLM_Raw_Response, Used_State_Update_Mode] = request_scene_json_from_llm(apiEndpoint, options, Model_Name, Current_User_Text, false);
        catch 
            error("Failed to establish cognitive connection with the local language model. Verify the system daemon is actively running and the network ports are securely accessible.");
        end
        
        agent_think_stream(think_area, "Intercepted communication from the language model. Decoding the structured data payload to extract physical variables.");
        [Simulation_Params, json_str, decode_success] = decode_scene_json_response(LLM_Raw_Response);
        
        if ~decode_success && Used_State_Update_Mode
            agent_think_stream(think_area, "The state aware dialogue refinement response was not decodable. Falling back to the full conversational history request path.");
            try
                [LLM_Raw_Response, Used_State_Update_Mode] = request_scene_json_from_llm(apiEndpoint, options, Model_Name, Current_User_Text, true);
                [Simulation_Params, json_str, decode_success] = decode_scene_json_response(LLM_Raw_Response);
            catch
                error("Fallback connection to the local language model failed securely.");
            end
        end

        Chat_History(end+1) = struct('role', 'assistant', 'content', LLM_Raw_Response, 'images', {cell(1,0)});

        if ~decode_success
            agent_think_stream(think_area, "Encountered a critical structural failure while attempting to decode the format of the response. Printing the raw output text below for physical diagnostics.");
            agent_think_stream(think_area, json_str);
            error("The language model did not return a valid structured format.");
        end
        
        Default_Params = struct('fc', 77e9, 'tp', 1e-3, 'B', 4e9, 'PRF', 4000, ...
            'fs', 10e6, 'tx_pos', [0, 5, 1], 'rx_pos', [0, 5, 1], 'antenna_gain', 15, ...
            'antenna_beamwidth', 60, 'antenna_isolation', 40, 'SNR', 50, ...
            'enable_wall', false, 'enable_multipath', true, ...
            'wall_center', [0, 1, 1], 'wall_dimensions', [3, 0.2, 2.5], ...
            'wall_epsilon_r', 4.5, 'wall_loss_tangent', 0.05, ...
            'stft_window_seconds', 0.1, 'stft_overlap_ratio', 0.75, 'enable_network', false);

        if Used_State_Update_Mode && Has_Latest_State
            Simulation_Params = merge_simulation_state(Latest_State_Params, Simulation_Params, Default_Params);
        else
            fields_list = fieldnames(Default_Params);
            for idx_field = 1:length(fields_list)
                if ~isfield(Simulation_Params, fields_list{idx_field})
                    Simulation_Params.(fields_list{idx_field}) = Default_Params.(fields_list{idx_field});
                end
            end
        end
        
        Simulation_Params.tx_pos = parse_antenna_positions(Simulation_Params.tx_pos);
        Simulation_Params.rx_pos = parse_antenna_positions(Simulation_Params.rx_pos);

        if isfield(Simulation_Params, 'persons')
            if iscell(Simulation_Params.persons)
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
                Simulation_Params.persons = Simulation_Params;
            else
                error("The language model did not return a valid persons multi-target array.");
            end
        end
        
        num_persons = length(Simulation_Params.persons);
        for i_p = 1:num_persons
            if ~isfield(Simulation_Params.persons(i_p), 'start_pos') || isempty(Simulation_Params.persons(i_p).start_pos), Simulation_Params.persons(i_p).start_pos = [0, 0, 0]; end
            if ~isfield(Simulation_Params.persons(i_p), 'start_heading') || isempty(Simulation_Params.persons(i_p).start_heading), Simulation_Params.persons(i_p).start_heading = 0.0; end
            if ~isfield(Simulation_Params.persons(i_p), 'start_time_delay') || isempty(Simulation_Params.persons(i_p).start_time_delay), Simulation_Params.persons(i_p).start_time_delay = 0.0; end
            if ~isfield(Simulation_Params.persons(i_p), 'height') || isempty(Simulation_Params.persons(i_p).height), Simulation_Params.persons(i_p).height = 1.70; end
            if ~isfield(Simulation_Params.persons(i_p), 'weight') || isempty(Simulation_Params.persons(i_p).weight), Simulation_Params.persons(i_p).weight = 70.0; end
        end

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

        Simulation_Params = sanitize_simulation_params(Simulation_Params, Default_Params);
        if Used_State_Update_Mode && Has_Latest_State
            Simulation_Params = apply_incremental_dialogue_adjustments(Simulation_Params, Latest_State_Params, Current_User_Text);
        end
        num_persons = length(Simulation_Params.persons);

        agent_think_stream(think_area, sprintf("Successfully extracted MIMO hardware properties, refined %d semantic motion prompts and constructed %d environmental object geometries natively.", num_persons, length(Simulation_Params.objects)));
        
        original_dir = pwd;
        if ~isfolder(EMDM_Path)
            error("The specified EMDM motion generation directory does not exist within the current execution environment.");
        end
        cd(EMDM_Path);
        dir_guard = onCleanup(@() cd(original_dir));
        
        for i_p = 1:num_persons
            agent_think_stream(think_area, sprintf("Delegating the three dimensional human motion trajectory generation for Target %d to the deep diffusion mathematical framework.", i_p));

            Prompt_Candidates = build_emdm_prompt_candidates(Simulation_Params.persons(i_p), Current_User_Text);
            latest_npy = "";
            status_gen = -1;
            cmdout_gen = "";

            model_pth = fullfile('models', 'HumanML3D.pth');
            for attempt_idx = 1:length(Prompt_Candidates)
                Cleaned_Prompt = string(Prompt_Candidates(attempt_idx));
                Cleaned_Prompt = regexprep(Cleaned_Prompt, '[\r\n]+', ' ');
                Cleaned_Prompt = strrep(Cleaned_Prompt, '"', '');
                Cleaned_Prompt = strrep(Cleaned_Prompt, '''', '');

                pre_run_files = dir(fullfile('models', '**', 'results.npy'));
                pre_run_paths = string(fullfile({pre_run_files.folder}, {pre_run_files.name}));
                run_start_time = now;

                cmd = sprintf('conda run --no-capture-output -n %s python sample_mdm.py --text_prompt "%s" --model_path "%s" --dataset humanml --guidance_param %s', ...
                              Conda_Env, Cleaned_Prompt, model_pth, Guidance_Param);

                if attempt_idx == 1
                    agent_think_stream(think_area, sprintf("Executing system shell command for Target %d. Triggering the external python inference kernel.", i_p));
                else
                    agent_think_stream(think_area, sprintf("The previous motion phrase for Target %d did not yield a valid result. Retrying with a stabilized prompt candidate %d...", i_p, attempt_idx));
                end
                [status_gen, cmdout_gen] = system(cmd);

                latest_npy = select_latest_results_file(pre_run_paths, fullfile('models', '**', 'results.npy'), run_start_time);
                if strlength(latest_npy) > 0
                    Simulation_Params.persons(i_p).Resolved_Prompt = Cleaned_Prompt;
                    break;
                end
            end

            if strlength(latest_npy) == 0
                if strlength(string(cmdout_gen)) > 0
                    fprintf('%s\n', cmdout_gen);
                end
                agent_think_stream(think_area, sprintf("Action generation failed for Target %d. No valid resulting matrix was detected within the system pipeline.", i_p));
                error("Action generation failed. No new or updated resulting array was generated. Skipping visualization.");
            elseif status_gen ~= 0
                agent_think_stream(think_area, sprintf("The external motion generation environment returned a non zero status for Target %d, but a valid resulting array was still detected and will be used.", i_p));
            end

            % Allocate unique temporary file identifiers preventing access collisions systematically across concurrent threads
            unique_hex_id = sprintf('%08X', randi([0, intmax('uint32')]));
            temp_py = fullfile(tempdir, sprintf('RadHARAGT_py_%s_%d.py', unique_hex_id, i_p));
            temp_mat = fullfile(tempdir, sprintf('RadHARAGT_mat_%s_%d.mat', unique_hex_id, i_p));
            
            cleanup_py = onCleanup(@() delete_if_exists(temp_py));
            cleanup_mat = onCleanup(@() delete_if_exists(temp_mat));

            fid_py = fopen(temp_py, 'w');
            if fid_py < 0
                error("Unable to create the temporary python conversion script bridging motion matrices securely.");
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

            latest_npy_char = char(latest_npy);
            conv_cmd = sprintf('conda run --no-capture-output -n %s python "%s" "%s" "%s"', Conda_Env, temp_py, latest_npy_char, temp_mat);
            [status_conv, cmdout_conv] = system(conv_cmd);

            if ~isfile(temp_mat)
                if status_conv ~= 0, fprintf('%s\n', cmdout_conv); end
                agent_think_stream(think_area, "Data format conversion pipeline collapsed. Python process failed to bridge external structures.");
                error("Python conversion failed to create the target temporary data file. Please investigate python and deep learning environments.");
            elseif status_conv ~= 0
                agent_think_stream(think_area, sprintf("The temporary numpy to matlab conversion routine for Target %d emitted a non zero status, but the converted matrix was generated successfully and will be used.", i_p));
            end

            loaded_data = load(temp_mat);
            if ~isfield(loaded_data, 'motion')
                error("The converted temporary data file does not contain a readable motion matrix.");
            end
            motion_mat = sanitize_motion_matrix(double(loaded_data.motion));
            
            clear cleanup_py cleanup_mat; 

            agent_think_stream(think_area, sprintf("Applying kinematic smoothing, orientation mappings, spatial translations and temporal bounds correctly to Target %d.", i_p));

            window_size = robust_sg_window(size(motion_mat, 1), 9);
            poly_order = min(3, max(window_size - 2, 1));
            if window_size >= 5 && poly_order < window_size
                for j = 1:size(motion_mat, 2)
                    for d = 1:size(motion_mat, 3)
                        if all(isfinite(motion_mat(:, j, d)))
                            motion_mat(:, j, d) = sgolayfilt(motion_mat(:, j, d), poly_order, window_size);
                        end
                    end
                end
            end

            orig_frames = size(motion_mat, 1);
            root_pos = motion_mat(:, 1, :);
            rel_motion = motion_mat - root_pos;
            rel_vel = diff(rel_motion, 1, 1); 

            frame_activity = squeeze(sum(sum(rel_vel.^2, 2), 3));
            if isempty(frame_activity) || ~any(isfinite(frame_activity))
                frame_activity = zeros(max(orig_frames - 1, 1), 1);
            end
            idle_threshold = max(1e-5, 0.02 * max(frame_activity)); 
            is_active = frame_activity > idle_threshold;

            last_active_frame = orig_frames;
            for f = orig_frames - 1 : -1 : 1
                if is_active(f)
                    last_active_frame = f + 1;
                    break;
                end
            end

            cut_frame = min(last_active_frame + 5, orig_frames);
            if cut_frame < orig_frames - 10 
                motion_mat = motion_mat(1:cut_frame, :, :);
            end

            root_x0 = motion_mat(1, 1, 1);
            root_y0 = motion_mat(1, 1, 3); 

            yaw_rad = deg2rad(mod(Simulation_Params.persons(i_p).start_heading, 360));
            start_pos = ensure_row_vector(Simulation_Params.persons(i_p).start_pos, [0, 0, 0], 3);

            rot_ang = yaw_rad - pi/2;
            cos_r = cos(rot_ang);
            sin_r = sin(rot_ang);

            x_centered = motion_mat(:, :, 1) - root_x0;
            y_centered = motion_mat(:, :, 3) - root_y0;
            motion_mat(:, :, 1) = x_centered * cos_r - y_centered * sin_r + start_pos(1);
            motion_mat(:, :, 3) = x_centered * sin_r + y_centered * cos_r + start_pos(2);
            motion_mat(:, :, 2) = motion_mat(:, :, 2) + start_pos(3);

            % Enforce precise ground alignment guaranteeing lowest kinematic extremities touch the physical floor coordinate natively preventing subterranean anomalies
            lowest_z = min(motion_mat(:, :, 2), [], 'all');
            floor_offset = max(0.0, 0.0 - lowest_z);
            motion_mat(:, :, 2) = motion_mat(:, :, 2) + floor_offset;

            delay_sec = max(0.0, double(Simulation_Params.persons(i_p).start_time_delay));
            delay_frames = round(delay_sec * Motion_FPS);
            if delay_frames > 0
                pad_front = repmat(motion_mat(1, :, :), [delay_frames, 1, 1]);
                motion_mat = cat(1, pad_front, motion_mat);
            end

            Simulation_Params.persons(i_p).collision_radius = estimate_person_footprint_radius(motion_mat);
            Simulation_Params.persons(i_p).motion_mat = motion_mat;
        end

        clear dir_guard;
        cd(original_dir); 
        
        agent_think_stream(think_area, "Aligning disparate temporal length trajectories across all detected physical entities appending steady states.");
        
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
        
        agent_think_stream(think_area, "Evaluating unified temporal physical bounds to calculate dynamic collision models preventing intersecting human kinematics and geometric environment penetrations.");
        
        wall_margin = 0.1;                                                      
        wall_bbox = [-inf, inf, -inf, inf];
        if Simulation_Params.enable_wall
            w_c = Simulation_Params.wall_center;
            w_d = Simulation_Params.wall_dimensions;
            wall_bbox = [w_c(1) - w_d(1)/2, w_c(1) + w_d(1)/2, w_c(2) - w_d(2)/2, w_c(2) + w_d(2)/2];
        end

        object_bboxes = zeros(0, 4);
        if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
            for i_o = 1:length(Simulation_Params.objects)
                sp_matrix = Simulation_Params.objects(i_o).scatter_points;
                if ~isempty(sp_matrix) && size(sp_matrix, 2) >= 3
                    object_bboxes(end + 1, :) = [min(sp_matrix(:, 1)), max(sp_matrix(:, 1)), min(sp_matrix(:, 2)), max(sp_matrix(:, 2))];
                end
            end
        end

        person_radius = zeros(1, num_persons);
        for i_p = 1:num_persons
            if isfield(Simulation_Params.persons(i_p), 'collision_radius') && isfinite(Simulation_Params.persons(i_p).collision_radius)
                person_radius(i_p) = Simulation_Params.persons(i_p).collision_radius;
            else
                person_radius(i_p) = estimate_person_footprint_radius(Simulation_Params.persons(i_p).motion_mat);
            end
        end

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
        
        for i_p = 1:num_persons
            if initial_off_X(i_p) ~= 0 || initial_off_Y(i_p) ~= 0
                Simulation_Params.persons(i_p).motion_mat(:, :, 1) = Simulation_Params.persons(i_p).motion_mat(:, :, 1) + initial_off_X(i_p);
                Simulation_Params.persons(i_p).motion_mat(:, :, 3) = Simulation_Params.persons(i_p).motion_mat(:, :, 3) + initial_off_Y(i_p);
            end
        end

        offset_X = zeros(max_frames, num_persons);
        offset_Y = zeros(max_frames, num_persons);

        for f = 1:max_frames
            if f > 1
                offset_X(f, :) = offset_X(f - 1, :);
                offset_Y(f, :) = offset_Y(f - 1, :);
            end

            for iter = 1:6
                curr_roots_X = zeros(num_persons, 1);
                curr_roots_Y = zeros(num_persons, 1);
                for i_p = 1:num_persons
                    curr_roots_X(i_p) = Simulation_Params.persons(i_p).motion_mat(f, 1, 1) + offset_X(f, i_p);
                    curr_roots_Y(i_p) = Simulation_Params.persons(i_p).motion_mat(f, 1, 3) + offset_Y(f, i_p);
                end

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

                for i_p = 1:num_persons
                    for j_p = i_p + 1:num_persons
                        xi = curr_roots_X(i_p);
                        yi = curr_roots_Y(i_p);
                        xj = curr_roots_X(j_p);
                        yj = curr_roots_Y(j_p);

                        vec = [xi - xj, yi - yj];
                        dist = norm(vec);
                        person_min_dist = person_radius(i_p) + person_radius(j_p);

                        soft_repulsion_factor = 0.45;
                        if dist < person_min_dist && dist > 1e-4
                            overlap = person_min_dist - dist;
                            push = (vec / dist) * (overlap * soft_repulsion_factor);

                            offset_X(f, i_p) = offset_X(f, i_p) + push(1);
                            offset_Y(f, i_p) = offset_Y(f, i_p) + push(2);
                            offset_X(f, j_p) = offset_X(f, j_p) - push(1);
                            offset_Y(f, j_p) = offset_Y(f, j_p) - push(2);
                        elseif dist <= 1e-4
                            seed_angle = 2 * pi * (i_p + j_p + f) / max(num_persons + max_frames, 1);
                            push = [cos(seed_angle), sin(seed_angle)] * (person_min_dist * soft_repulsion_factor);

                            offset_X(f, i_p) = offset_X(f, i_p) + push(1);
                            offset_Y(f, i_p) = offset_Y(f, i_p) + push(2);
                            offset_X(f, j_p) = offset_X(f, j_p) - push(1);
                            offset_Y(f, j_p) = offset_Y(f, j_p) - push(2);
                        end

                        curr_roots_X(i_p) = Simulation_Params.persons(i_p).motion_mat(f, 1, 1) + offset_X(f, i_p);
                        curr_roots_Y(i_p) = Simulation_Params.persons(i_p).motion_mat(f, 1, 3) + offset_Y(f, i_p);
                        curr_roots_X(j_p) = Simulation_Params.persons(j_p).motion_mat(f, 1, 1) + offset_X(f, j_p);
                        curr_roots_Y(j_p) = Simulation_Params.persons(j_p).motion_mat(f, 1, 3) + offset_Y(f, j_p);
                    end
                end
            end
        end

        agent_think_stream(think_area, "Applying temporal Savitzky-Golay filters universally harmonizing collision avoidance shifts to completely prevent unnatural micro-Doppler kinematic anomalies.");
        smooth_window = min(31, max_frames - mod(max_frames, 2) - 1);
        if smooth_window >= 5
            for i_p = 1:num_persons
                offset_X(:, i_p) = sgolayfilt(offset_X(:, i_p), 3, smooth_window);
                offset_Y(:, i_p) = sgolayfilt(offset_Y(:, i_p), 3, smooth_window);
            end
        end
        
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

        for i_p = 1:num_persons
            off_x = reshape(offset_X(:, i_p), [max_frames, 1, 1]);
            off_y = reshape(offset_Y(:, i_p), [max_frames, 1, 1]);
            
            Simulation_Params.persons(i_p).motion_mat(:, :, 1) = Simulation_Params.persons(i_p).motion_mat(:, :, 1) + off_x;
            Simulation_Params.persons(i_p).motion_mat(:, :, 3) = Simulation_Params.persons(i_p).motion_mat(:, :, 3) + off_y;
        end

        agent_think_stream(think_area, "Consolidating the extracted hardware properties, target kinematics and RCS properties into the persistent physical parameter collection system.");
        
        Simulation_Params.num_frames = max_frames;            
        Simulation_Params.num_joints = 22;            
        
        Simulation_Params.FPS = Motion_FPS;                            
        Simulation_Params.time_axis = (0 : Simulation_Params.num_frames - 1) / Motion_FPS; 
        
        kinematic_tree =[
            1, 2; 1, 3; 1, 4; 2, 5; 3, 6; 4, 7; 5, 8; 6, 9; 
            7, 10; 8, 11; 9, 12; 10, 13; 10, 14; 10, 15; 13, 16; 
            14, 17; 15, 18; 17, 19; 18, 20; 19, 21; 20, 22
        ];
        Simulation_Params.kinematic_tree = kinematic_tree;
        
        Normalized_RCS =[1.0, 0.7, 0.7, 0.9, 0.5, 0.5, 0.8, 0.3, 0.3, 0.8, ...
                          0.1, 0.1, 0.6, 0.5, 0.5, 0.85, 0.4, 0.4, 0.3, 0.3, ...
                          0.1, 0.1];
        Simulation_Params.Normalized_RCS = Normalized_RCS;
        
        min_joint_radius = 0.02;
        max_joint_radius = 0.075;
        joint_radii = min_joint_radius + (max_joint_radius - min_joint_radius) * Normalized_RCS;
        
        Simulation_Params.joint_radii = joint_radii;                   
        Simulation_Params.bone_radius = 0.035;                         
        
        base_hw_factor = sqrt(1.70 * 70.0);
        for i_p = 1:num_persons
            H = Simulation_Params.persons(i_p).height;
            W = Simulation_Params.persons(i_p).weight;
            RCS_scale = sqrt(H * W) / base_hw_factor;
            
            Simulation_Params.persons(i_p).RCS_scale = RCS_scale;
            Simulation_Params.persons(i_p).Scaled_RCS = Simulation_Params.Normalized_RCS * RCS_scale;
            
            radius_scale = sqrt(RCS_scale);
            Simulation_Params.persons(i_p).joint_radii = Simulation_Params.joint_radii * radius_scale;
            Simulation_Params.persons(i_p).bone_radius = Simulation_Params.bone_radius * radius_scale;
        end
        
        try
            save(fullfile(Save_Dir, 'Simulation_Params.mat'), 'Simulation_Params');
        catch
            agent_think_stream(think_area, "Permission constraints prevented writing the vast parameter file cleanly to disk. Retaining the active variable structures strictly in dynamic memory.");
        end
        Latest_State_Params = export_dialogue_state(Simulation_Params);
        Latest_State_JSON = string(jsonencode(Latest_State_Params));
        Has_Latest_State = true;

        agent_think_stream(think_area, "Engaging the frequency modulated continuous wave radar physics engine to computationally simulate superimposed point scatterers and complex wall interactions across multiple targets and MIMO channels simultaneously.");
        
        c = 3e8;                                            
        fc = Simulation_Params.fc;
        lambda = c / fc; 
        tp = Simulation_Params.tp;
        B = Simulation_Params.B;
        K = B / tp;                                         
        PRF = Simulation_Params.PRF;
        fs = Simulation_Params.fs;
        
        if isfield(Simulation_Params, 'antenna_gain')
            G_lin = 10^(Simulation_Params.antenna_gain / 10);
        else
            G_lin = 10^(10 / 10); 
        end
        
        base_amp_factor = (lambda * G_lin) / ((4 * pi)^1.5);
        beamwidth_rad = deg2rad(Simulation_Params.antenna_beamwidth);
        
        epsilon_0 = 8.854e-12;
        mu_0 = 4 * pi * 1e-7;
        omega = 2 * pi * fc;

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
            
            wall_alpha = omega * sqrt(mu_0 * eps_real / 2 * (sqrt(1 + (eps_imag/eps_real)^2) - 1));
            v_wall = c / sqrt(eps_r);
            wall_gamma = (1 - sqrt(eps_r)) / (1 + sqrt(eps_r));
            
            wall_y_center = Simulation_Params.wall_center(2);
            wall_y_thickness = Simulation_Params.wall_dimensions(2);
            wall_y_min = wall_y_center - wall_y_thickness / 2;
            wall_y_max = wall_y_center + wall_y_thickness / 2;
        end
        
        tx_pos_list = Simulation_Params.tx_pos;              
        rx_pos_list = Simulation_Params.rx_pos;
        num_tx = size(tx_pos_list, 1);
        num_rx = size(rx_pos_list, 1);
        num_channels = num_tx * num_rx;
        
        N_s = max(1, round(tp * fs));                               
        t_fast = (0:N_s-1)' / fs;
        
        t_motion = Simulation_Params.time_axis;             
        t_slow = 0 : 1/PRF : t_motion(end);                 
        num_pulses = length(t_slow);

        % Enforce strict operational memory limits automatically scaling parameters maintaining physical proportional ratios systematically
        Max_Elements_Per_Channel = 50e6; 
        Estimated_Elements = N_s * num_pulses;
        if Estimated_Elements > Max_Elements_Per_Channel
            agent_think_stream(think_area, "The requested spatial temporal configurations exceed standard memory bounds natively. Automatically downscaling the fast and slow time frequency sampling parameters securely preserving proportional physical kinematics systematically.");
            scale_factor = sqrt(Max_Elements_Per_Channel / Estimated_Elements);
            fs = max(1e4, round(fs * scale_factor));
            PRF = max(100, round(PRF * scale_factor));
            N_s = max(1, round(tp * fs));
            t_fast = (0:N_s-1)' / fs;
            t_slow = 0 : 1/PRF : t_motion(end);
            num_pulses = length(t_slow);
        end
        
        motion_radar_all = zeros(num_pulses, num_persons, 22, 3);
        for i_p = 1:num_persons
            motion_mat = Simulation_Params.persons(i_p).motion_mat;
            for j = 1:22
                motion_radar_all(:, i_p, j, 1) = interp1(t_motion, motion_mat(:,j,1), t_slow, 'pchip', 'extrap'); 
                motion_radar_all(:, i_p, j, 2) = interp1(t_motion, motion_mat(:,j,3), t_slow, 'pchip', 'extrap'); 
                motion_radar_all(:, i_p, j, 3) = interp1(t_motion, motion_mat(:,j,2), t_slow, 'pchip', 'extrap'); 
            end
        end
        
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
        
        static_echo_matrix = complex(zeros(N_s, num_channels));
        if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
            agent_think_stream(think_area, "Precomputing highly dense static background electromagnetic scattering clusters preserving exact coherent phase arrays efficiently.");
            
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
                    rng(1000 + i_o);
                    obj_random_phases = 2 * pi * rand(num_sp, 1);
                    
                    for p_idx = 1:num_sp
                        target_pos = sp_matrix(p_idx, 1:3);
                        current_rcs_val = sp_matrix(p_idx, 4);
                        
                        R_tx = norm(target_pos - tx_pos_curr);
                        R_rx = norm(target_pos - rx_pos_curr);
                        R_total = R_tx + R_rx;
                        
                        v_tx = (target_pos - tx_pos_curr) / (R_tx + 1e-6);
                        v_rx = (target_pos - rx_pos_curr) / (R_rx + 1e-6);
                        
                        % Establish broad radiation projection mapping realistic indoor patch antenna behaviors naturally
                        theta_tx = acos(max(min(dot(v_tx, tx_boresight_curr), 1), -1));
                        theta_rx = acos(max(min(dot(v_rx, rx_boresight_curr), 1), -1));
                        pat_tx = exp(-1.386 * (theta_tx / beamwidth_rad)^2);
                        pat_rx = exp(-1.386 * (theta_rx / beamwidth_rad)^2);
                        pattern_factor = pat_tx * pat_rx;
                        
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
                            theta_tx_mpA = acos(max(min(dot(v_tx_mpA, tx_boresight_curr), 1), -1));
                            pat_tx_mpA = exp(-1.386 * (theta_tx_mpA / beamwidth_rad)^2);
                            pattern_factor_A = pat_tx_mpA * pat_rx;
                            tau_mpA = (R_tx_mp + R_rx) / c + tau_tx_mp_wall_delay + tau_rx_wall_delay;
                            amp_loss_A = trans_loss_tx_mp * trans_loss_rx * exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall));
                            amp_mpA = gamma_gnd * base_amp_factor * sqrt(current_rcs_val) / (R_tx_mp * R_rx + 1e-6) * amp_loss_A * pattern_factor_A; 
                            phase_mpA = 2*pi*(fc*tau_mpA + K*tau_mpA*t_fast - 0.5*K*tau_mpA^2) + obj_random_phases(p_idx);
                            signal_static_ch = signal_static_ch + amp_mpA .* exp(1i * phase_mpA);
                            
                            v_rx_mpB = (target_pos_mp - rx_pos_curr) / (R_rx_mp + 1e-6);
                            theta_rx_mpB = acos(max(min(dot(v_rx_mpB, rx_boresight_curr), 1), -1));
                            pat_rx_mpB = exp(-1.386 * (theta_rx_mpB / beamwidth_rad)^2);
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
                                    theta_tx_wall_mp = acos(max(min(dot(v_tx_wall_mp, tx_boresight_curr), 1), -1));
                                    pat_tx_wall_mp = exp(-1.386 * (theta_tx_wall_mp / beamwidth_rad)^2);
                                    pattern_factor_2A = pat_tx_wall_mp * pat_rx;
                                    tau_mp2A = (R_tx_wall_mp + R_rx) / c + tau_rx_wall_delay; 
                                    amp_mp2A = wall_gamma * base_amp_factor * sqrt(current_rcs_val) / (R_tx_wall_mp * R_rx + 1e-6) * trans_loss_rx * exp(-wall_alpha * d_rx_wall) * pattern_factor_2A;
                                    phase_mp2A = 2*pi*(fc*tau_mp2A + K*tau_mp2A*t_fast - 0.5*K*tau_mp2A^2) + obj_random_phases(p_idx);
                                    signal_static_ch = signal_static_ch + amp_mp2A .* exp(1i * phase_mp2A);
                                end
                                
                                if is_reflection_path_valid(rx_pos_curr, target_pos_wall_mp, target_wall_face, Simulation_Params.wall_center, Simulation_Params.wall_dimensions)
                                    v_rx_wall_mp = (target_pos_wall_mp - rx_pos_curr) / (R_rx_wall_mp + 1e-6);
                                    theta_rx_wall_mp = acos(max(min(dot(v_rx_wall_mp, rx_boresight_curr), 1), -1));
                                    pat_rx_wall_mp = exp(-1.386 * (theta_rx_wall_mp / beamwidth_rad)^2);
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

        raw_echo = complex(zeros(N_s, num_pulses, num_channels));
        
        rng(Random_Seed); 
        joint_random_phases = 2 * pi * rand(num_persons, 22);

        num_M = num_persons * 22;
        target_pos_flat = zeros(num_pulses, num_M, 3);
        target_rcs_flat_base = zeros(1, num_M);
        target_phase_flat = zeros(1, num_M);
        
        idx = 1;
        for i_p = 1:num_persons
            for j = 1:22
                target_pos_flat(:, idx, 1) = motion_radar_all(:, i_p, j, 1);
                target_pos_flat(:, idx, 2) = motion_radar_all(:, i_p, j, 3);
                target_pos_flat(:, idx, 3) = motion_radar_all(:, i_p, j, 2);
                target_rcs_flat_base(idx) = Simulation_Params.persons(i_p).Scaled_RCS(j);
                target_phase_flat(idx) = joint_random_phases(i_p, j);
                idx = idx + 1;
            end
        end

        % Apply dynamic temporal structural fluctuations projecting realistic aspect variation traces uniformly
        fluctuation_phases = 2 * pi * rand(1, num_M);
        fluctuation_rates = 0.05 + 0.15 * rand(1, num_M);
        fluctuation_matrix = 1.0 + 0.3 * sin(2 * pi * t_slow' * fluctuation_rates + fluctuation_phases);
        target_rcs_dynamic = target_rcs_flat_base .* fluctuation_matrix;
        
        N_total = num_pulses * num_M;
        target_pos_flat = reshape(target_pos_flat, N_total, 3);
        target_rcs_flat = target_rcs_dynamic(:);
        target_phase_flat = repmat(target_phase_flat, num_pulses, 1);
        target_phase_flat = target_phase_flat(:);

        agent_think_stream(think_area, sprintf("Calculating multi-target MIMO bistatic radar equations linearly propagating signals over %d spatial channels. Advanced matrix vectorization engaged.", num_channels));
        
        for ch = 1:num_channels
            tx_idx = ceil(ch / num_rx);
            rx_idx = mod(ch - 1, num_rx) + 1;
            tx_pos_curr = tx_pos_list(tx_idx, :);
            rx_pos_curr = rx_pos_list(rx_idx, :);
            tx_boresight_curr = tx_boresight_list(tx_idx, :);
            rx_boresight_curr = rx_boresight_list(rx_idx, :);
            
            Path_Tau = {};
            Path_Amp = {};
            Path_Phase = {};

            target_vec_tx = target_pos_flat - tx_pos_curr;
            R_tx = sqrt(sum(target_vec_tx.^2, 2));
            target_vec_rx = target_pos_flat - rx_pos_curr;
            R_rx = sqrt(sum(target_vec_rx.^2, 2));
            R_total = R_tx + R_rx;

            v_tx = target_vec_tx ./ (R_tx + 1e-6);
            v_rx = target_vec_rx ./ (R_rx + 1e-6);
            theta_tx = acos(max(min(sum(v_tx .* tx_boresight_curr, 2), 1), -1));
            theta_rx = acos(max(min(sum(v_rx .* rx_boresight_curr, 2), 1), -1));
            pat_tx = exp(-1.386 * (theta_tx / beamwidth_rad).^2);
            pat_rx = exp(-1.386 * (theta_rx / beamwidth_rad).^2);
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
                theta_tx_mpA = acos(max(min(sum(v_tx_mpA .* tx_boresight_curr, 2), 1), -1));
                pat_tx_mpA = exp(-1.386 * (theta_tx_mpA / beamwidth_rad).^2);
                pattern_factor_A = pat_tx_mpA .* pat_rx;
                
                tau_mpA = (R_tx_mp + R_rx) / c + tau_tx_mp_wall_delay + tau_rx_wall_delay;
                amp_loss_A = trans_loss_tx_mp .* trans_loss_rx .* exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall));
                amp_mpA = gamma_gnd * base_amp_factor * sqrt(target_rcs_flat) ./ (R_tx_mp .* R_rx + 1e-6) .* amp_loss_A .* pattern_factor_A; 
                
                Path_Tau{end+1} = reshape(tau_mpA, num_pulses, num_M);
                Path_Amp{end+1} = reshape(amp_mpA, num_pulses, num_M);
                Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

                v_rx_mpB = (target_pos_mp - rx_pos_curr) ./ (R_rx_mp + 1e-6);
                theta_rx_mpB = acos(max(min(sum(v_rx_mpB .* rx_boresight_curr, 2), 1), -1));
                pat_rx_mpB = exp(-1.386 * (theta_rx_mpB / beamwidth_rad).^2);
                pattern_factor_B = pat_tx .* pat_rx_mpB;
                
                tau_mpB = (R_tx + R_rx_mp) / c + tau_tx_wall_delay + tau_rx_mp_wall_delay;
                amp_loss_B = trans_loss_tx .* trans_loss_rx_mp .* exp(-wall_alpha * (d_tx_wall + d_rx_wall_mp));
                amp_mpB = gamma_gnd * base_amp_factor * sqrt(target_rcs_flat) ./ (R_tx .* R_rx_mp + 1e-6) .* amp_loss_B .* pattern_factor_B; 
                
                Path_Tau{end+1} = reshape(tau_mpB, num_pulses, num_M);
                Path_Amp{end+1} = reshape(amp_mpB, num_pulses, num_M);
                Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

                pattern_factor_C = pat_tx_mpA .* pat_rx_mpB;
                tau_mpC = (R_tx_mp + R_rx_mp) / c + tau_tx_mp_wall_delay + tau_rx_mp_wall_delay;
                amp_loss_C = trans_loss_tx_mp .* trans_loss_rx_mp .* exp(-wall_alpha * (d_tx_wall_mp + d_rx_wall_mp));
                amp_mpC = (gamma_gnd^2) * base_amp_factor * sqrt(target_rcs_flat) ./ (R_tx_mp .* R_rx_mp + 1e-6) .* amp_loss_C .* pattern_factor_C; 
                
                Path_Tau{end+1} = reshape(tau_mpC, num_pulses, num_M);
                Path_Amp{end+1} = reshape(amp_mpC, num_pulses, num_M);
                Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);
                
                if Simulation_Params.enable_wall
                    target_wall_face = zeros(N_total, 1);
                    idx_top = target_pos_flat(:, 2) > wall_y_max;
                    target_wall_face(idx_top) = wall_y_max;
                    target_wall_face(~idx_top) = wall_y_min;
                    
                    target_pos_wall_mp = target_pos_flat;
                    target_pos_wall_mp(:, 2) = 2 * target_wall_face - target_pos_flat(:, 2);
                    
                    R_tx_wall_mp = sqrt(sum((target_pos_wall_mp - tx_pos_curr).^2, 2));
                    R_rx_wall_mp = sqrt(sum((target_pos_wall_mp - rx_pos_curr).^2, 2));
                    
                    valid_2A = is_reflection_path_valid_vec(tx_pos_curr, target_pos_wall_mp, target_wall_face, Simulation_Params.wall_center, Simulation_Params.wall_dimensions);
                    v_tx_wall_mp = (target_pos_wall_mp - tx_pos_curr) ./ (R_tx_wall_mp + 1e-6);
                    theta_tx_wall_mp = acos(max(min(sum(v_tx_wall_mp .* tx_boresight_curr, 2), 1), -1));
                    pat_tx_wall_mp = exp(-1.386 * (theta_tx_wall_mp / beamwidth_rad).^2);
                    pattern_factor_2A = pat_tx_wall_mp .* pat_rx;
                    
                    tau_mp2A = (R_tx_wall_mp + R_rx) / c + tau_rx_wall_delay;
                    amp_mp2A = zeros(N_total, 1);
                    amp_mp2A(valid_2A) = wall_gamma * base_amp_factor * sqrt(target_rcs_flat(valid_2A)) ./ (R_tx_wall_mp(valid_2A) .* R_rx(valid_2A) + 1e-6) .* trans_loss_rx(valid_2A) .* exp(-wall_alpha * d_rx_wall(valid_2A)) .* pattern_factor_2A(valid_2A);
                    
                    Path_Tau{end+1} = reshape(tau_mp2A, num_pulses, num_M);
                    Path_Amp{end+1} = reshape(amp_mp2A, num_pulses, num_M);
                    Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

                    valid_2B = is_reflection_path_valid_vec(rx_pos_curr, target_pos_wall_mp, target_wall_face, Simulation_Params.wall_center, Simulation_Params.wall_dimensions);
                    v_rx_wall_mp = (target_pos_wall_mp - rx_pos_curr) ./ (R_rx_wall_mp + 1e-6);
                    theta_rx_wall_mp = acos(max(min(sum(v_rx_wall_mp .* rx_boresight_curr, 2), 1), -1));
                    pat_rx_wall_mp = exp(-1.386 * (theta_rx_wall_mp / beamwidth_rad).^2);
                    pattern_factor_2B = pat_tx .* pat_rx_wall_mp;
                    
                    tau_mp2B = (R_tx + R_rx_wall_mp) / c + tau_tx_wall_delay;
                    amp_mp2B = zeros(N_total, 1);
                    amp_mp2B(valid_2B) = wall_gamma * base_amp_factor * sqrt(target_rcs_flat(valid_2B)) ./ (R_tx(valid_2B) .* R_rx_wall_mp(valid_2B) + 1e-6) .* trans_loss_tx(valid_2B) .* exp(-wall_alpha * d_tx_wall(valid_2B)) .* pattern_factor_2B(valid_2B);
                    
                    Path_Tau{end+1} = reshape(tau_mp2B, num_pulses, num_M);
                    Path_Amp{end+1} = reshape(amp_mp2B, num_pulses, num_M);
                    Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);

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
                        tau_mp4 = zeros(N_total, 1);
                        tau_mp4(valid_4) = tau_base(valid_4) + 2 * D_tw(valid_4) / c;
                        amp_mp4 = zeros(N_total, 1);
                        amp_mp4(valid_4) = amp_base(valid_4) .* abs(wall_gamma) .* (R_total(valid_4) ./ (R_total(valid_4) + 2 * D_tw(valid_4))) * 0.5;
                        
                        Path_Tau{end+1} = reshape(tau_mp4, num_pulses, num_M);
                        Path_Amp{end+1} = reshape(amp_mp4, num_pulses, num_M);
                        Path_Phase{end+1} = reshape(target_phase_flat, num_pulses, num_M);
                        
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

            R_leakage = max(norm(tx_pos_curr - rx_pos_curr), 1e-3); 
            tau_leakage = R_leakage / c;
            if isfield(Simulation_Params, 'antenna_isolation')
                iso_lin = 10^(-Simulation_Params.antenna_isolation / 20);
            else
                iso_lin = 10^(-40 / 20);
            end
            amp_leakage = (lambda * G_lin) / (4 * pi * R_leakage) * iso_lin;
            phase_leakage = 2*pi*(fc*tau_leakage + K*tau_leakage*t_fast - 0.5*K*tau_leakage^2);

            num_paths = length(Path_Tau);
            for p = 1:num_pulses
                signal_p = complex(zeros(N_s, 1));
                signal_p = signal_p + amp_leakage .* exp(1i * phase_leakage) + static_echo_matrix(:, ch);
                
                for path_idx = 1:num_paths
                    tau_p = Path_Tau{path_idx}(p, :);
                    amp_p = Path_Amp{path_idx}(p, :);
                    phase_flat_p = Path_Phase{path_idx}(p, :);
                    
                    valid = amp_p > 1e-12;
                    if any(valid)
                        tau_v = tau_p(valid);
                        amp_v = amp_p(valid);
                        phase_v = phase_flat_p(valid);
                        
                        phase_mat = 2*pi*(fc*tau_v + K*t_fast*tau_v - 0.5*K*tau_v.^2) + phase_v;
                        signal_p = signal_p + sum(amp_v .* exp(1i * phase_mat), 2);
                    end
                end
                raw_echo(:, p, ch) = signal_p;
            end
        end
        
        ref_distance = 2.0;
        ref_amp = (lambda * G_lin * 1.0) / ((4*pi)^1.5 * ref_distance^2);
        ref_power = ref_amp^2;
        noise_power = (ref_power / (10^(Simulation_Params.SNR / 10))) * N_s;
        
        noise_matrix = sqrt(noise_power/2) * (randn(size(raw_echo)) + 1i * randn(size(raw_echo)));
        raw_echo = raw_echo + noise_matrix;
        
        try
            save(fullfile(Save_Dir, 'Raw_Echo.mat'), 'raw_echo', '-v7.3');
        catch
            agent_think_stream(think_area, "Permission constraints prevented writing the vast matrix file cleanly to disk. Retaining the active echo arrays strictly in dynamic memory.");
        end
        
        agent_think_stream(think_area, "Executing discrete numerical fourier transforms upon the simulated MIMO echoes resolving spatial range coordinates natively propagating variables dynamically.");
        
        fast_time_window = hamming(N_s);
        raw_echo_windowed = raw_echo .* fast_time_window;
        range_fft = fft(raw_echo_windowed, N_s, 1);
        RTM_complex_all = range_fft(1:floor(N_s/2), :, :);         
        
        f_beat = (0:floor(N_s/2)-1)' * (fs / N_s);
        range_axis = f_beat * c / (2 * K);

        max_indoor_range = 25.0;                                                
        valid_range_idx = range_axis <= max_indoor_range;
        RTM_complex_all = RTM_complex_all(valid_range_idx, :, :);
        range_axis = range_axis(valid_range_idx);
        
        clear range_fft raw_echo_windowed raw_echo noise_matrix;
        
        agent_think_stream(think_area, "Applying Moving Target Indication universally subtracting exact static backgrounds isolating purely kinetic micro-Doppler signatures seamlessly.");
        RTM_complex_all = RTM_complex_all - mean(RTM_complex_all, 2);
        
        STC_Weights = (range_axis.^2) + 1e-3;
        max_stc_gain = 250.0;
        [~, ref_idx] = min(abs(range_axis - 2.0));
        STC_Weights = STC_Weights / STC_Weights(max(1, ref_idx)); 
        STC_Weights = min(STC_Weights, max_stc_gain);
        RTM_complex_all = RTM_complex_all .* STC_Weights;

        RTM_Mag_Lin_All = abs(RTM_complex_all);
        RTM_Mag_Lin_First = RTM_Mag_Lin_All(:, :, 1);
        RTM_Mag_Lin_Sum = sum(RTM_Mag_Lin_All, 3) / num_channels; 
        
        range_variance = var(RTM_Mag_Lin_Sum, 0, 2);
        active_threshold = max(range_variance) * 0.05; 
        active_bins = find(range_variance > active_threshold);
        
        if isempty(active_bins)
            [~, max_idx] = max(mean(RTM_Mag_Lin_Sum, 2));
            active_bins = max_idx;
        end
        
        plot_r_min = max(0, range_axis(active_bins(1)) - 2.0);
        plot_r_max = min(max_indoor_range, range_axis(active_bins(end)) + 8.0);
        
        n_window = min(num_pulses, max(round(Simulation_Params.stft_window_seconds * PRF), 8));
        if mod(n_window, 2) == 0 && n_window > 1
            n_window = n_window - 1;
        end
        n_window = max(n_window, min(num_pulses, 5));
        n_overlap = min(round(n_window * Simulation_Params.stft_overlap_ratio), max(n_window - 1, 0));
        nfft_stft = min(4096, max(256, 2^nextpow2(max(n_window, 32))));
        [~, F_doppler, time_stft] = spectrogram(RTM_complex_all(active_bins(1), :, 1), hamming(n_window), n_overlap, nfft_stft, PRF, 'centered');
        
        DTM_Energy_First = zeros(length(F_doppler), length(time_stft));
        DTM_Energy_Sum = zeros(length(F_doppler), length(time_stft));
        
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
        
        DTM_Energy_Sum = DTM_Energy_Sum / num_channels;
        
        % Extract the linear magnitude matrices strictly preserving the ascending physical frequency sequence avoiding inverted Doppler visualizations
        DTM_Mag_Lin_First = sqrt(DTM_Energy_First);
        DTM_Mag_Lin_Sum = sqrt(DTM_Energy_Sum);

        agent_think_stream(think_area, "Normalizing matrix magnitudes independently and converting to rigorous logarithmic decibel scales representing multiple channel representations seamlessly.");
        
        RTM_Mag_Log_First = 20*log10(RTM_Mag_Lin_First + 1e-12);
        DTM_Mag_Log_First = 20*log10(DTM_Mag_Lin_First + 1e-12);
        
        RTM_Mag_Log_Sum = 20*log10(RTM_Mag_Lin_Sum + 1e-12);
        DTM_Mag_Log_Sum = 20*log10(DTM_Mag_Lin_Sum + 1e-12);
        
        RTM_Mag_Log_First_Norm = normalize_log_matrix_securely(RTM_Mag_Log_First);
        DTM_Mag_Log_First_Norm = normalize_log_matrix_securely(DTM_Mag_Log_First);
        
        RTM_Mag_Log_Sum_Norm = normalize_log_matrix_securely(RTM_Mag_Log_Sum);
        DTM_Mag_Log_Sum_Norm = normalize_log_matrix_securely(DTM_Mag_Log_Sum);

        if Simulation_Params.enable_network
            agent_think_stream(think_area, "Connecting the local deep learning denoising framework to systematically eliminate background static while preserving dynamic kinematic MIMO channel traces.");
            try
                net = denoisingNetwork('dncnn');
                
                denoise_matrix = @(Mat) ...
                    double(denoiseImage(single((Mat - min(Mat(:))) / (max(Mat(:)) - min(Mat(:)) + 1e-12)), net)) * (max(Mat(:)) - min(Mat(:))) + min(Mat(:));
                
                RTM_Mag_Log_First_Norm = denoise_matrix(RTM_Mag_Log_First_Norm);
                DTM_Mag_Log_First_Norm = denoise_matrix(DTM_Mag_Log_First_Norm);
                
                RTM_Mag_Log_Sum_Norm = denoise_matrix(RTM_Mag_Log_Sum_Norm);
                DTM_Mag_Log_Sum_Norm = denoise_matrix(DTM_Mag_Log_Sum_Norm);
                
            catch
                agent_think_stream(think_area, "The deep neural network enhancement mechanism produced an internal operational failure. Terminating the refinement routine to maintain spectral stability.");
            end
        end

        agent_think_stream(think_area, "Formulating completely distinct First Channel and Channel Sum compressed high fidelity image matrices and exporting to the drive in strict png format natively.");
        
        log_clim_min = -35;
        log_clim_max = 0;
        
        RTM_First_Log_Clamped = max(min(RTM_Mag_Log_First_Norm, log_clim_max), log_clim_min);
        DTM_First_Log_Clamped = max(min(DTM_Mag_Log_First_Norm, log_clim_max), log_clim_min);
        
        RTM_Sum_Log_Clamped = max(min(RTM_Mag_Log_Sum_Norm, log_clim_max), log_clim_min);
        DTM_Sum_Log_Clamped = max(min(DTM_Mag_Log_Sum_Norm, log_clim_max), log_clim_min);
        
        custom_cmap_256 = interp1(linspace(0, 1, size(JoeyBG_Colormap_Flip, 1)), JoeyBG_Colormap_Flip, linspace(0, 1, 256));
        
        % Implement a universal orientation inversion applying standard Cartesian projection mapping lowest values to the bottom edge preserving intuitive analytical layouts natively
        render_and_save_png = @(Matrix_Log, file_name) ...
            imwrite(imresize(ind2rgb(gray2ind(normalize_log_image_for_export(flipud(Matrix_Log), log_clim_min, log_clim_max), 256), custom_cmap_256), [1024, 1024]), fullfile(Save_Dir, file_name));

        render_and_save_png(RTM_First_Log_Clamped, 'RTM_First_Channel_Log_1024x1024.png');
        render_and_save_png(DTM_First_Log_Clamped, 'DTM_First_Channel_Log_1024x1024.png');
        render_and_save_png(RTM_Sum_Log_Clamped,   'RTM_Channel_Sum_Log_1024x1024.png');
        render_and_save_png(DTM_Sum_Log_Clamped,   'DTM_Channel_Sum_Log_1024x1024.png');

        agent_think_stream(think_area, "Generating comprehensive multithreaded graphical interface components displaying four requested separate structures mapping First Channel and Channel Sum explicitly.");
        
        [ax_rtm1, ax_dtm1, ax_rtms, ax_dtms, ax_3d] = append_result_bubble();

        local_spectrum_plotter('RTM (First Channel, Log)', t_slow, range_axis, RTM_Mag_Log_First_Norm, ax_rtm1, [plot_r_min, plot_r_max], 'Range (m)', Font_Name, Font_Size_Basis, Font_Weight_Basis, Font_Size_Title, Font_Weight_Title, Font_Size_Axis, Font_Weight_Axis, JoeyBG_Colormap_Flip);
        local_spectrum_plotter('DTM (First Channel, Log)', time_stft, F_doppler, DTM_Mag_Log_First_Norm, ax_dtm1, [-PRF/2, PRF/2], 'Doppler (Hz)', Font_Name, Font_Size_Basis, Font_Weight_Basis, Font_Size_Title, Font_Weight_Title, Font_Size_Axis, Font_Weight_Axis, JoeyBG_Colormap_Flip);
        local_spectrum_plotter('RTM (Channel Sum, Log)', t_slow, range_axis, RTM_Mag_Log_Sum_Norm, ax_rtms, [plot_r_min, plot_r_max], 'Range (m)', Font_Name, Font_Size_Basis, Font_Weight_Basis, Font_Size_Title, Font_Weight_Title, Font_Size_Axis, Font_Weight_Axis, JoeyBG_Colormap_Flip);
        local_spectrum_plotter('DTM (Channel Sum, Log)', time_stft, F_doppler, DTM_Mag_Log_Sum_Norm, ax_dtms, [-PRF/2, PRF/2], 'Doppler (Hz)', Font_Name, Font_Size_Basis, Font_Weight_Basis, Font_Size_Title, Font_Weight_Title, Font_Size_Axis, Font_Weight_Axis, JoeyBG_Colormap_Flip);

        agent_think_stream(think_area, "Constructing the primary three dimensional workspace rendering pipeline to project and animate the physical motions and structural scene layouts mapping all active independent antenna clusters natively.");
        
        frames = Simulation_Params.num_frames;
        
        hold(ax_3d, 'on');
        grid(ax_3d, 'on');
        
        set(ax_3d, 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Basis, 'Box', 'on', 'LineWidth', 1.2, 'GridAlpha', 0.25, 'GridColor', [0.4 0.4 0.4]);
        title_obj = title(ax_3d, 'Motion Visualization', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title, 'Color', [0.15 0.15 0.15]);
        xlabel(ax_3d, 'X (Azimuth, m)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis, 'Color', [0.25 0.25 0.25]);
        ylabel(ax_3d, 'Y (Range, m)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis, 'Color', [0.25 0.25 0.25]);
        zlabel(ax_3d, 'Z (Height, m)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis, 'Color', [0.25 0.25 0.25]);
        ax_3d.XColor = [0.2 0.2 0.2];
        ax_3d.YColor = [0.2 0.2 0.2];
        ax_3d.ZColor = [0.2 0.2 0.2];
        
        view(ax_3d, 45, 20);
        colormap(ax_3d, JoeyBG_Colormap_Flip);
        clim(ax_3d,[0, 1]); 
        
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

        xlim(ax_3d,[x_min, x_max]);
        ylim(ax_3d,[y_min, y_max]);
        zlim(ax_3d,[0, z_max]);
        
        grid_step = 0.15; 
        x_grid_vals = floor(x_min):grid_step:ceil(x_max);
        y_grid_vals = floor(y_min):grid_step:ceil(y_max);
        [GX, GY] = meshgrid(x_grid_vals, y_grid_vals);
        GZ = zeros(size(GX));
        
        surf(GX, GY, GZ, 'Parent', ax_3d, ...
             'FaceColor',[0.25 0.25 0.25], 'FaceAlpha', 0.6, ...
             'EdgeColor', [0.15 0.15 0.15], 'EdgeAlpha', 0.5, ...
             'HandleVisibility', 'off');
             
        for t = 1:num_tx
            if t == 1
                plot3(ax_3d, tx_pos_list(t, 1), tx_pos_list(t, 2), tx_pos_list(t, 3), '^', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(2,:), 'MarkerEdgeColor', [0.1 0.1 0.1], 'LineWidth', 1.5, 'DisplayName', 'TX Antenna(s)');
            else
                plot3(ax_3d, tx_pos_list(t, 1), tx_pos_list(t, 2), tx_pos_list(t, 3), '^', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(2,:), 'MarkerEdgeColor', [0.1 0.1 0.1], 'LineWidth', 1.5, 'HandleVisibility', 'off');
            end
        end
        
        for r = 1:num_rx
            if r == 1
                plot3(ax_3d, rx_pos_list(r, 1), rx_pos_list(r, 2), rx_pos_list(r, 3), 'v', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(end-1,:), 'MarkerEdgeColor', [0.1 0.1 0.1], 'LineWidth', 1.5, 'DisplayName', 'RX Antenna(s)');
            else
                plot3(ax_3d, rx_pos_list(r, 1), rx_pos_list(r, 2), rx_pos_list(r, 3), 'v', 'MarkerSize', 12, 'MarkerFaceColor', JoeyBG_Colormap(end-1,:), 'MarkerEdgeColor', [0.1 0.1 0.1], 'LineWidth', 1.5, 'HandleVisibility', 'off');
            end
        end
        
        if Simulation_Params.enable_wall
            wc = Simulation_Params.wall_center;
            wd = Simulation_Params.wall_dimensions;
            
            X_w = wc(1) + [-1 1 1 -1 -1 1 1 -1] * wd(1)/2;
            Y_w = wc(2) +[-1 -1 1 1 -1 -1 1 1] * wd(2)/2;
            Z_w = wc(3) +[-1 -1 -1 -1 1 1 1 1] * wd(3)/2;
            
            faces =[1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
            patch('Vertices',[X_w' Y_w' Z_w'], 'Faces', faces, 'Parent', ax_3d, ...
                  'FaceColor',[0.70 0.75 0.85], 'FaceAlpha', 0.25, 'EdgeColor', [0.40 0.45 0.55], ...
                  'EdgeAlpha', 0.6, 'LineWidth', 1.5, 'DisplayName', 'Simulation Wall');
        end

        if isfield(Simulation_Params, 'objects') && ~isempty(Simulation_Params.objects)
            agent_think_stream(think_area, "Rendering the reconstructed simple static objects mapping complex geometries via dense structural scattering clusters in three dimensional space.");
            
            plot3(ax_3d, nan, nan, nan, 's', 'MarkerFaceColor', JoeyBG_Colormap(9, :), 'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'DisplayName', 'Static Scene Objects');
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
                        sc_size = max(5, min(20, sqrt(sc_rcs) * 12));
                        
                        plot3(ax_3d, sp_matrix(i_sp, 1), sp_matrix(i_sp, 2), sp_matrix(i_sp, 3), 's', ...
                              'MarkerSize', sc_size, 'MarkerFaceColor', JoeyBG_Colormap(9, :), ...
                              'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
                    end
                    
                    if size(sp_matrix, 1) >= 4
                        try
                            K_hull = convhull(sp_matrix(:,1), sp_matrix(:,2), sp_matrix(:,3));
                            trisurf(K_hull, sp_matrix(:,1), sp_matrix(:,2), sp_matrix(:,3), 'Parent', ax_3d, 'FaceColor', JoeyBG_Colormap(9,:), 'FaceAlpha', 0.15, 'EdgeColor', JoeyBG_Colormap(9,:), 'EdgeAlpha', 0.3, 'HandleVisibility', 'off');
                        catch
                        end
                    end
                end
            end
        end
              
        camlight(ax_3d, 'headlight');
        camlight(ax_3d, 'left');
        lighting(ax_3d, 'gouraud');
        material(ax_3d, 'dull'); 
        
        rep_joint_color = JoeyBG_Colormap(4, :);
        rep_bone_color = JoeyBG_Colormap_Flip(4, :);
        
        plot3(ax_3d, nan, nan, nan, 'o', 'MarkerFaceColor', rep_joint_color, 'MarkerEdgeColor', 'none', 'MarkerSize', 10, 'DisplayName', 'Human Joints');
        plot3(ax_3d, nan, nan, nan, '-', 'Color', rep_bone_color, 'LineWidth', 4, 'DisplayName', 'Human Bones');
        plot3(ax_3d, nan, nan, nan, '-', 'Color', [rep_joint_color, 0.4], 'LineWidth', 1.5, 'DisplayName', 'Joint Trajectories');
        legend(ax_3d, 'Location', 'northeast', 'FontName', Font_Name, 'FontSize', Font_Size_Basis-6);

        [cyl_x, cyl_y, cyl_z] = cylinder([1.0, 0.65], 20); 
        [sph_x, sph_y, sph_z] = sphere(20);                
        
        num_joints = Simulation_Params.num_joints;
        num_bones = size(kinematic_tree, 1);
        cmap_len = size(JoeyBG_Colormap, 1);
        
        rcs_color_indices = round((1 - Normalized_RCS) * (cmap_len - 1)) + 1;
        rcs_color_indices = max(min(rcs_color_indices, cmap_len), 1); 
        
        h_joints_t = gobjects(num_persons, num_joints);
        h_bones_t = gobjects(num_persons, num_bones);
        h_trajs = gobjects(num_persons, num_joints);
        
        traj_X = cell(num_persons, 1);
        traj_Y = cell(num_persons, 1);
        traj_Z = cell(num_persons, 1);
        
        for i_p = 1:num_persons
            traj_X{i_p} = nan(frames, num_joints);
            traj_Y{i_p} = nan(frames, num_joints);
            traj_Z{i_p} = nan(frames, num_joints);
            
            for j = 1:num_joints
                rcs_idx = rcs_color_indices(j);
                joint_color = JoeyBG_Colormap(rcs_idx, :);
                
                h_joints_t(i_p, j) = hgtransform('Parent', ax_3d);
                surf(sph_x, sph_y, sph_z, 'Parent', h_joints_t(i_p, j), 'FaceColor', joint_color, ...
                     'EdgeColor', 'none', 'HandleVisibility', 'off', ...
                     'SpecularStrength', 0.2, 'DiffuseStrength', 0.8);
                     
                h_trajs(i_p, j) = plot3(ax_3d, nan, nan, nan, 'Color',[joint_color, 0.35], ...
                                   'LineWidth', 1.5, 'HandleVisibility', 'off');
            end
            
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
                
                h_bones_t(i_p, b) = hgtransform('Parent', ax_3d);
                
                surf(cyl_x, cyl_y, cyl_z, bone_cdata, 'Parent', h_bones_t(i_p, b), ...
                     'FaceColor', 'interp', ...
                     'EdgeColor', 'none', 'HandleVisibility', 'off', ...
                     'SpecularStrength', 0.1, 'DiffuseStrength', 0.9);
            end
        end

        agent_think_stream(think_area, "Synchronizing internal plot rendering loops to animate multi target joints along their mathematically resolved spatial vectors natively rendering distinct sizes.");
        
        target_frame_duration = 1.0 / Simulation_Params.FPS;
        for f = 1:frames
            loop_start = tic;
            if ~isgraphics(ax_3d)
                break; 
            end
            
            title_obj.String = sprintf('Motion Visualization at %.2f Seconds', (f-1)/Simulation_Params.FPS);
            
            for i_p = 1:num_persons
                curr_joints = squeeze(Simulation_Params.persons(i_p).motion_mat(f, :, :));
                
                X_data = curr_joints(:, 1);
                Y_data = curr_joints(:, 3); 
                Z_data = curr_joints(:, 2); 
                
                traj_X{i_p}(f, :) = X_data';
                traj_Y{i_p}(f, :) = Y_data';
                traj_Z{i_p}(f, :) = Z_data';
                
                for j = 1:num_joints
                    h_trajs(i_p, j).XData = traj_X{i_p}(1:f, j);
                    h_trajs(i_p, j).YData = traj_Y{i_p}(1:f, j);
                    h_trajs(i_p, j).ZData = traj_Z{i_p}(1:f, j);
                end
                
                for j = 1:num_joints
                    T = makehgtform('translate',[X_data(j), Y_data(j), Z_data(j)]);
                    S = makehgtform('scale', Simulation_Params.persons(i_p).joint_radii(j));
                    h_joints_t(i_p, j).Matrix = T * S;
                end
                
                for b = 1:num_bones
                    j1 = kinematic_tree(b, 1);
                    j2 = kinematic_tree(b, 2);
                    
                    p1 =[X_data(j1), Y_data(j1), Z_data(j1)];
                    p2 =[X_data(j2), Y_data(j2), Z_data(j2)];
                    
                    h_bones_t(i_p, b).Matrix = compute_bone_matrix(p1, p2, Simulation_Params.persons(i_p).bone_radius);
                end
            end
            
            drawnow limitrate;
            elapsed_time = toc(loop_start);
            pause_duration = max(0.001, target_frame_duration - elapsed_time);
            pause(pause_duration);
        end
        
        agent_think_stream(think_area, "The entire complex sequence of multi-person mathematical simulation formulations and distinct structural MIMO visual rendering processes has been successfully concluded.");
    end

    %% Internal Nested Helper Functions Subroutines natively isolating calculations
    function delete_if_exists(filepath)
        if isfile(filepath)
            try
                delete(filepath);
            catch
            end
        end
    end

    function [LLM_Raw_Response, Used_State_Update_Mode] = request_scene_json_from_llm(apiEndpoint, options, Model_Name, Current_User_Text, force_full_history)
        Used_State_Update_Mode = false;

        if nargin < 5
            force_full_history = false;
        end

        if ~force_full_history && Has_Latest_State && is_incremental_update_request(Current_User_Text)
            Used_State_Update_Mode = true;
            update_messages = build_state_update_messages(Current_User_Text);
            payload = struct('model', Model_Name, 'messages', update_messages, 'stream', false);
        else
            payload = struct('model', Model_Name, 'messages', Chat_History, 'stream', false);
        end

        try
            response = webwrite(apiEndpoint, payload, options);
            LLM_Raw_Response = response.message.content;
        catch
            error("Failed to establish cognitive connection with the local language model. Verify the system daemon is actively running and the network ports are securely accessible.");
        end
    end

    function update_messages = build_state_update_messages(Current_User_Text)
        State_Update_System_Prompt = strcat(...
            "你现在是一个专门负责多轮场景修订的JSON状态维护器。", ...
            "你会得到上一轮已经成功仿真的完整JSON状态，以及用户本轮追加的修改指令。", ...
            "你的任务是输出一份新的完整JSON。未被用户明确修改的字段必须严格保持不变。", ...
            "如果用户只修改人物速度、方向、起点、延迟或姿态，只允许修改必要字段。", ...
            "Refined_Prompt 必须写成稳定自然的 HumanML3D 风格英文动作短句，只描述动作类别、速度和姿态。", ...
            "不要在 Refined_Prompt 中写坐标轴、绝对方向、toward x axis、toward y axis、reverse x axis、clockwise around axis、雷达、墙体、桌子等场景信息。", ...
            "人物空间朝向的改变应优先通过 start_heading 表达。若用户要求朝反方向走，优先将 start_heading 在上一轮基础上加 180 度并规范到 0 到 360，Refined_Prompt 仍保持自然 walking 描述。", ...
            "输出必须且只能是一个合法JSON对象，不要输出解释。");

        update_payload = strcat(...
            "上一轮成功场景JSON如下：", char(Latest_State_JSON), "", ...
            "新的用户修改指令如下：", char(Current_User_Text), "", ...
            "请输出更新后的完整JSON。");

        update_messages = [ ...
            struct('role', 'system', 'content', State_Update_System_Prompt, 'images', {cell(1,0)}), ...
            struct('role', 'user', 'content', update_payload, 'images', {cell(1,0)}) ...
        ];
    end

    function [Simulation_Params, json_str, decode_success] = decode_scene_json_response(LLM_Raw_Response)
        json_str = strtrim(string(LLM_Raw_Response));
        idx_first = strfind(json_str, '{');
        idx_last = strfind(json_str, '}');
        if ~isempty(idx_first) && ~isempty(idx_last)
            json_str = extractBetween(json_str, idx_first(1), idx_last(end), 'Boundaries', 'inclusive');
        else
            json_str = regexprep(json_str, '^```json\s*', '');
            json_str = regexprep(json_str, '^```\s*', '');
            json_str = regexprep(json_str, '\s*```$', '');
        end

        json_str = clean_json_math(char(json_str));
        decode_success = true;
        try
            Simulation_Params = jsondecode(json_str);
        catch
            Simulation_Params = struct();
            decode_success = false;
        end
    end

    function is_incremental = is_incremental_update_request(Current_User_Text)
        text_lower = lower(strtrim(string(Current_User_Text)));
        if strlength(text_lower) == 0
            is_incremental = false;
            return;
        end

        update_keywords_cn = ["改", "修改", "调整", "加快", "减慢", "加速", "减速", "反方向", "掉头", "转身", "变成", "换成", "保持其他不变", "基础上", "追加", "往回走"];
        update_keywords_en = ["change", "modify", "adjust", "speed up", "slow down", "reverse", "opposite direction", "turn around", "keep others unchanged", "based on the previous"];
        full_scene_keywords = ["雷达", "radar", "prf", "bandwidth", "带宽", "天线", "antenna", "墙体", "wall", "桌子", "table", "objects", "fc", "fs", "tx_pos", "rx_pos"];

        has_update_keyword = any(contains(text_lower, lower(update_keywords_cn))) || any(contains(text_lower, update_keywords_en));
        has_full_scene_keyword = any(contains(text_lower, full_scene_keywords));
        is_short_instruction = strlength(text_lower) <= 80;

        is_incremental = has_update_keyword && (is_short_instruction || ~has_full_scene_keyword);
    end

    function merged_params = merge_simulation_state(previous_params, new_params, Default_Params)
        merged_params = previous_params;
        if ~isstruct(merged_params) || isempty(fieldnames(merged_params))
            merged_params = Default_Params;
        end

        if isstruct(new_params)
            top_fields = fieldnames(new_params);
            for idx_field = 1:length(top_fields)
                merged_params.(top_fields{idx_field}) = new_params.(top_fields{idx_field});
            end
        end

        if ~isfield(merged_params, 'persons') || isempty(merged_params.persons)
            if isfield(new_params, 'persons') && ~isempty(new_params.persons)
                merged_params.persons = new_params.persons;
            elseif isfield(previous_params, 'persons') && ~isempty(previous_params.persons)
                merged_params.persons = previous_params.persons;
            else
                merged_params.persons = struct('Refined_Prompt', "a person stands naturally and remains still", 'start_pos', [0 0 0], 'start_heading', 0.0, 'start_time_delay', 0.0, 'height', 1.70, 'weight', 70.0);
            end
        end

        if ~isfield(merged_params, 'objects')
            merged_params.objects = [];
        end

        if isfield(merged_params, 'persons') && ~isempty(merged_params.persons)
            merged_params.persons = merge_person_arrays(get_struct_value(previous_params, 'persons', []), merged_params.persons);
        end

        if isfield(merged_params, 'objects') && ~isempty(merged_params.objects)
            merged_params.objects = merge_object_arrays(get_struct_value(previous_params, 'objects', []), merged_params.objects);
        end

        default_fields = fieldnames(Default_Params);
        for idx_field = 1:length(default_fields)
            if ~isfield(merged_params, default_fields{idx_field}) || isempty(merged_params.(default_fields{idx_field}))
                merged_params.(default_fields{idx_field}) = Default_Params.(default_fields{idx_field});
            end
        end
    end

    function merged_persons = merge_person_arrays(previous_persons, new_persons)
        if iscell(new_persons)
            new_persons = cell_to_struct_array(new_persons);
        end
        if isempty(previous_persons)
            merged_persons = new_persons;
            return;
        end
        if iscell(previous_persons)
            previous_persons = cell_to_struct_array(previous_persons);
        end

        merged_persons = new_persons;
        for idx_person = 1:length(merged_persons)
            if idx_person <= length(previous_persons)
                prev_fields = fieldnames(previous_persons(idx_person));
                for idx_field = 1:length(prev_fields)
                    if ~isfield(merged_persons(idx_person), prev_fields{idx_field}) || isempty(merged_persons(idx_person).(prev_fields{idx_field}))
                        merged_persons(idx_person).(prev_fields{idx_field}) = previous_persons(idx_person).(prev_fields{idx_field});
                    end
                end
            end
        end
    end

    function merged_objects = merge_object_arrays(previous_objects, new_objects)
        if iscell(new_objects)
            new_objects = cell_to_struct_array(new_objects);
        end
        if isempty(previous_objects)
            merged_objects = new_objects;
            return;
        end
        if iscell(previous_objects)
            previous_objects = cell_to_struct_array(previous_objects);
        end

        merged_objects = new_objects;
        for idx_object = 1:length(merged_objects)
            if idx_object <= length(previous_objects)
                prev_fields = fieldnames(previous_objects(idx_object));
                for idx_field = 1:length(prev_fields)
                    if ~isfield(merged_objects(idx_object), prev_fields{idx_field}) || isempty(merged_objects(idx_object).(prev_fields{idx_field}))
                        merged_objects(idx_object).(prev_fields{idx_field}) = previous_objects(idx_object).(prev_fields{idx_field});
                    end
                end
            end
        end
    end

    function struct_array = cell_to_struct_array(input_cell)
        if isempty(input_cell)
            struct_array = struct([]);
            return;
        end

        all_fields = {};
        for idx_item = 1:length(input_cell)
            if isstruct(input_cell{idx_item})
                all_fields = union(all_fields, fieldnames(input_cell{idx_item}));
            end
        end

        struct_array = struct();
        for idx_item = 1:length(input_cell)
            for idx_field = 1:length(all_fields)
                if isfield(input_cell{idx_item}, all_fields{idx_field})
                    struct_array(idx_item).(all_fields{idx_field}) = input_cell{idx_item}.(all_fields{idx_field});
                else
                    struct_array(idx_item).(all_fields{idx_field}) = [];
                end
            end
        end
    end

    function Simulation_Params = apply_incremental_dialogue_adjustments(Simulation_Params, Previous_Params, Current_User_Text)
        if ~isfield(Simulation_Params, 'persons') || isempty(Simulation_Params.persons)
            return;
        end

        user_lower = lower(string(Current_User_Text));
        reverse_requested = contains(user_lower, "反方向") || contains(user_lower, "往回") || contains(user_lower, "掉头") || contains(user_lower, "turn around") || contains(user_lower, "opposite direction") || contains(user_lower, "reverse");
        faster_requested = contains(user_lower, "加快") || contains(user_lower, "加速") || contains(user_lower, "更快") || contains(user_lower, "faster") || contains(user_lower, "speed up") || contains(user_lower, "quicker");
        slower_requested = contains(user_lower, "减慢") || contains(user_lower, "更慢") || contains(user_lower, "slow down") || contains(user_lower, "slower");

        for idx_person = 1:length(Simulation_Params.persons)
            previous_heading = 0.0;
            if isfield(Previous_Params, 'persons') && idx_person <= length(Previous_Params.persons) && isfield(Previous_Params.persons(idx_person), 'start_heading')
                previous_heading = double(Previous_Params.persons(idx_person).start_heading);
            end

            if reverse_requested
                new_heading = double(Simulation_Params.persons(idx_person).start_heading);
                if abs(mod(new_heading - previous_heading, 360)) < 45 || abs(mod(new_heading - previous_heading, 360) - 360) < 45
                    Simulation_Params.persons(idx_person).start_heading = mod(previous_heading + 180.0, 360.0);
                end
            end

            raw_prompt = "";
            if isfield(Simulation_Params.persons(idx_person), 'Refined_Prompt')
                raw_prompt = string(Simulation_Params.persons(idx_person).Refined_Prompt);
            end

            if faster_requested || slower_requested || reverse_requested
                Simulation_Params.persons(idx_person).Refined_Prompt = build_safe_motion_prompt(raw_prompt, Current_User_Text);
            end
        end
    end

    function prompt_candidates = build_emdm_prompt_candidates(person_struct, Current_User_Text)
        raw_prompt = "a person stands naturally and remains still";
        if isfield(person_struct, 'Refined_Prompt') && strlength(strtrim(string(person_struct.Refined_Prompt))) > 0
            raw_prompt = string(person_struct.Refined_Prompt);
        end

        candidate_list = strings(0, 1);
        candidate_list(end + 1, 1) = sanitize_emdm_prompt_text(raw_prompt);
        candidate_list(end + 1, 1) = build_safe_motion_prompt(raw_prompt, Current_User_Text);
        candidate_list(end + 1, 1) = build_safe_motion_prompt("", Current_User_Text);
        candidate_list(end + 1, 1) = "a person walking at a medium speed, maintaining upright posture and natural gait";
        candidate_list(end + 1, 1) = "a person stands naturally and remains still";

        candidate_list = candidate_list(strlength(strtrim(candidate_list)) > 0);
        prompt_candidates = unique(candidate_list, 'stable');
    end

    function safe_prompt = build_safe_motion_prompt(raw_prompt, Current_User_Text)
        combined_text = lower(string(raw_prompt) + " " + string(Current_User_Text));
        safe_prompt = "a person walking at a medium speed, maintaining upright posture and natural gait";

        if any(contains(combined_text, ["stand", "still", "静止", "站立"]))
            safe_prompt = "a person stands naturally and remains still";
            return;
        end
        if any(contains(combined_text, ["sit", "sits", "seated", "坐下", "坐着"]))
            safe_prompt = "a person sits down steadily and ends in a seated posture";
            return;
        end
        if any(contains(combined_text, ["fall", "跌倒", "摔倒"]))
            safe_prompt = "a person falls down quickly and ends lying on the ground";
            return;
        end
        if any(contains(combined_text, ["run", "running", "跑", "奔跑"]))
            if any(contains(combined_text, ["slow", "slower", "减慢", "更慢"]))
                safe_prompt = "a person jogging slowly with stable balance";
            elseif any(contains(combined_text, ["fast", "faster", "quick", "加快", "更快", "加速"]))
                safe_prompt = "a person running quickly with stable balance and natural posture";
            else
                safe_prompt = "a person running at a moderate speed with stable balance and natural posture";
            end
            return;
        end

        if any(contains(combined_text, ["slow", "slower", "减慢", "更慢", "缓慢"]))
            safe_prompt = "a person walking slowly, maintaining upright posture and natural gait";
        elseif any(contains(combined_text, ["fast", "faster", "quick", "quicker", "加快", "更快", "加速", "speed up"]))
            safe_prompt = "a person walking quickly, maintaining upright posture and natural gait";
        else
            safe_prompt = "a person walking at a medium speed, maintaining upright posture and natural gait";
        end
    end

    function clean_prompt = sanitize_emdm_prompt_text(raw_prompt)
        clean_prompt = string(raw_prompt);
        clean_prompt = regexprep(clean_prompt, '[\r\n]+', ' ');
        clean_prompt = strrep(clean_prompt, '"', '');
        clean_prompt = strrep(clean_prompt, '''', '');
        clean_prompt = regexprep(clean_prompt, '\s+', ' ');

        direction_patterns = { ...
            '(?i)towards?\s+the\s+[xyz]\s*axis', ...
            '(?i)towards?\s+[+-]?\s*[xy]\s*axis', ...
            '(?i)along\s+the\s+[xyz]\s*axis', ...
            '(?i)back\s+towards?\s+the\s+[xyz]\s*axis', ...
            '(?i)backwards?\s+towards?\s+.*?(?=,|\.|$)', ...
            '(?i)reverse\s+direction', ...
            '(?i)opposite\s+direction', ...
            '(?i)clockwise\s+around\s+.*?(?=,|\.|$)', ...
            '(?i)counterclockwise\s+around\s+.*?(?=,|\.|$)', ...
            '(?i)with\s+heading\s+.*?(?=,|\.|$)', ...
            '(?i)towards?\s+positive\s+[xy]', ...
            '(?i)towards?\s+negative\s+[xy]'};
        for idx_pattern = 1:length(direction_patterns)
            clean_prompt = regexprep(clean_prompt, direction_patterns{idx_pattern}, '');
        end
        clean_prompt = regexprep(clean_prompt, '(?i)backwards?', 'forward');
        clean_prompt = regexprep(clean_prompt, '\s+,', ',');
        clean_prompt = regexprep(clean_prompt, '\s+', ' ');
        clean_prompt = strtrim(clean_prompt);

        if strlength(clean_prompt) == 0
            clean_prompt = "a person walking at a medium speed, maintaining upright posture and natural gait";
        end
    end

    function dialogue_state = export_dialogue_state(Simulation_Params)
        dialogue_state = rmfield_if_present(Simulation_Params, {'num_frames', 'num_joints', 'FPS', 'time_axis', 'kinematic_tree', 'Normalized_RCS', 'joint_radii', 'bone_radius'});
        if isfield(dialogue_state, 'persons')
            dialogue_state.persons = rmfield_if_present(dialogue_state.persons, {'motion_mat', 'collision_radius', 'RCS_scale', 'Scaled_RCS', 'joint_radii', 'bone_radius', 'Resolved_Prompt'});
        end
    end

    function output_struct = rmfield_if_present(input_struct, field_names)
        output_struct = input_struct;
        for idx_field = 1:length(field_names)
            if isstruct(output_struct) && isfield(output_struct, field_names{idx_field})
                output_struct = rmfield(output_struct, field_names{idx_field});
            end
        end
    end

    function cleaned_str = clean_json_math(raw_str)
        cleaned_str = raw_str;
        % Remove stray alphabetic characters preceding property keys ensuring strictly formatted string sequences
        cleaned_str = regexprep(cleaned_str, '([\{\[,]\s*)[a-zA-Z0-9_]+"', '$1"');
        
        % Strip trailing commas positioned directly before structural closing brackets preserving array integrity
        cleaned_str = regexprep(cleaned_str, ',\s*([\]\}])', '$1');
        
        % Standardize boolean logic strings conforming correctly to standardized data definition cases
        cleaned_str = regexprep(cleaned_str, ':\s*True', ': true', 'ignorecase');
        cleaned_str = regexprep(cleaned_str, ':\s*False', ': false', 'ignorecase');
        
        % Iteratively parse and evaluate mathematical expressions substituting evaluated numeric results dynamically
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

    function mat_norm = normalize_log_matrix_securely(mat_in)
        max_val = max(mat_in(:));
        if isinf(max_val) || isnan(max_val)
            mat_norm = double(mat_in) * 0;
        else
            mat_norm = double(mat_in) - max_val;
        end
    end

    function local_spectrum_plotter(Title_Str, X_Axis, Y_Axis, Matrix, ax_temp, Y_Lims, Y_Label, Font_Name, Font_Size_Basis, Font_Weight_Basis, Font_Size_Title, Font_Weight_Title, Font_Size_Axis, Font_Weight_Axis, Cmap)
        imagesc(ax_temp, X_Axis, Y_Axis, Matrix);
        set(ax_temp, 'YDir', 'normal', 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis, 'Box', 'on', 'LineWidth', 1.2, 'TickDir', 'out');
        colormap(ax_temp, Cmap); 
        clim(ax_temp, [-35, 0]);
        title(ax_temp, Title_Str, 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title, 'Color', [0.15 0.15 0.15]);
        xlabel(ax_temp, 'Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis, 'Color', [0.25 0.25 0.25]);
        ylabel(ax_temp, Y_Label, 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis, 'Color', [0.25 0.25 0.25]);
        ylim(ax_temp, Y_Lims); 
        ax_temp.XColor = [0.2 0.2 0.2];
        ax_temp.YColor = [0.2 0.2 0.2];
    end

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

    function M = compute_bone_matrix(p1, p2, radius)
        v = p2 - p1;
        L = norm(v);
        
        if L < 1e-5
            M = makehgtform('scale', 0);
            return;
        end
        
        dir_vec = v / L;
        z_axis =[0, 0, 1];
        
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
        
        T = makehgtform('translate', p1);
        S = makehgtform('scale',[radius, radius, L]);
        
        M = T * R * S;
    end

    function Simulation_Params = sanitize_simulation_params(Simulation_Params, Default_Params)
        Simulation_Params.fc = clamp_scalar(get_struct_value(Simulation_Params, 'fc', Default_Params.fc), 1e8, 3e11, Default_Params.fc);
        Simulation_Params.tp = clamp_scalar(get_struct_value(Simulation_Params, 'tp', Default_Params.tp), 1e-6, 1.0, Default_Params.tp);
        Simulation_Params.B = clamp_scalar(get_struct_value(Simulation_Params, 'B', Default_Params.B), 1e6, 20e9, Default_Params.B);
        Simulation_Params.PRF = clamp_scalar(get_struct_value(Simulation_Params, 'PRF', Default_Params.PRF), 1.0, 2e5, Default_Params.PRF);
        Simulation_Params.fs = clamp_scalar(get_struct_value(Simulation_Params, 'fs', Default_Params.fs), 1e3, 2e8, Default_Params.fs);
        Simulation_Params.antenna_gain = clamp_scalar(get_struct_value(Simulation_Params, 'antenna_gain', Default_Params.antenna_gain), -10, 60, Default_Params.antenna_gain);
        Simulation_Params.antenna_beamwidth = clamp_scalar(get_struct_value(Simulation_Params, 'antenna_beamwidth', Default_Params.antenna_beamwidth), 10, 180, Default_Params.antenna_beamwidth);
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
    end

    function value = get_struct_value(input_struct, field_name, default_value)
        if isstruct(input_struct) && isfield(input_struct, field_name) && ~isempty(input_struct.(field_name))
            value = input_struct.(field_name);
        else
            value = default_value;
        end
    end

    function value = clamp_scalar(value_in, min_value, max_value, default_value)
        value = double(value_in);
        if isempty(value) || ~isscalar(value) || ~isfinite(value)
            value = default_value;
        end
        value = min(max(value, min_value), max_value);
    end

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

    function window_size = robust_sg_window(num_frames, requested_window)
        window_size = min(num_frames, requested_window);
        if mod(window_size, 2) == 0
            window_size = window_size - 1;
        end
        if window_size < 3
            window_size = 0;
        end
    end

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

    function [d_wall, tau_wall_delay, trans_loss_amp, gamma_eff] = compute_wall_path_terms_vec(src_pos, dst_pos_mat, wall_center, wall_dimensions, v_wall, c, eps_r, wall_gamma)
        P = size(dst_pos_mat, 1);
        d_wall = zeros(P, 1);
        tau_wall_delay = zeros(P, 1);
        trans_loss_amp = ones(P, 1);
        gamma_eff = wall_gamma * ones(P, 1);
        
        delta_vec = dst_pos_mat - src_pos; 
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

    function is_valid = is_reflection_path_valid_vec(src_pos, image_target_pos_mat, wall_face_y, wall_center, wall_dimensions)
        P = size(image_target_pos_mat, 1);
        is_valid = false(P, 1);
        
        if isscalar(wall_face_y)
            wall_face_y = repmat(wall_face_y, P, 1);
        end
        
        delta_vec = image_target_pos_mat - src_pos; 
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

    function gamma_eff = compute_dielectric_reflection_coeff(cos_theta, eps_r)
        cos_theta = min(max(cos_theta, 0.0), 1.0);
        eps_r = max(double(eps_r), 1.0);
        sin_theta_sq = max(0.0, 1.0 - cos_theta^2);
        root_term = sqrt(max(eps_r - sin_theta_sq, 0.0));

        gamma_te = (cos_theta - root_term) / (cos_theta + root_term + 1e-12);
        gamma_tm = (eps_r * cos_theta - root_term) / (eps_r * cos_theta + root_term + 1e-12);
        gamma_eff = 0.5 * (gamma_te + gamma_tm);
    end

    function normalized_img = normalize_log_image_for_export(Matrix_Log, log_clim_min, log_clim_max)
        normalized_img = (double(Matrix_Log) - log_clim_min) / max(log_clim_max - log_clim_min, 1e-12);
        normalized_img(~isfinite(normalized_img)) = 0;
        normalized_img = min(max(normalized_img, 0), 1);
    end
end