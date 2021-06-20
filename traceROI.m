function [polygons, labels, index] = traceROI(ima)
% TRACEROI - interactively trace polygonal Regions Of Interest (ROI) over an image
%
% [POLYGONS, LABELS, INDEX] = TRACEROI(IMA)  interactively trace polygonal
% Regions Of Interest (ROI) over the image IMA and returns the
% corresponding POLYGONS, LABELS and the pixel INDEX.
%
% Syntax:  [polygons, labels, index] = traceROI(ima)
%
% Inputs:
%    ima - MxNx3 numeric matrix, image over which polygons will be traced  
%
% Outputs:

%    polygons: U x 1 cell array containing the xy coordinates of each ROI polygon
%
%    labels: U x 1 vector containing the class label for each polygon
%
%    index: U x 1 cell array containing the corresponding pixel indices of each ROI polygon
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
% Compatibility: tested on Matlab R2016a
%
% Author: Matthew Parkan, EPFL - GIS Research Laboratory
% Website: http://lasig.epfl.ch/
% Last revision: November 2, 2016


%% setup GUI geometry

SCREEN_DIMS = get(0, 'ScreenSize');
SCREEN_WIDTH = SCREEN_DIMS(3);
SCREEN_HEIGHT = SCREEN_DIMS(4);
MIN_WINDOW_WIDTH = 600;
MIN_WINDOW_HEIGHT = 460;
WINDOW_LEFT_EDGE_POS = 0.25 * SCREEN_WIDTH;
WINDOW_BOTTOM_EDGE_POS = 0.1 * SCREEN_HEIGHT;
WINDOW_WIDTH = 0.55 * SCREEN_WIDTH;
WINDOW_HEIGHT = 0.7 * SCREEN_HEIGHT;
MARGIN = 10;
MARGIN_STATUSBAR = 10;

CONTROL_PANEL_WIDTH = 300;
CONTROL_PANEL_HEIGHT =  WINDOW_HEIGHT - (MARGIN + MARGIN_STATUSBAR); % WINDOW_HEIGHT - 2 * MARGIN;
CONTROL_PANEL_LEFT_EDGE_POS = WINDOW_WIDTH - CONTROL_PANEL_WIDTH - MARGIN;
CONTROL_PANEL_BOTTOM_EDGE_POS = MARGIN_STATUSBAR; % MARGIN;

VIEWER_PANEL_WIDTH = WINDOW_WIDTH - CONTROL_PANEL_WIDTH - 3 * MARGIN;
VIEWER_PANEL_HEIGHT =  WINDOW_HEIGHT - (MARGIN + MARGIN_STATUSBAR); % WINDOW_HEIGHT - 2 * MARGIN;
VIEWER_PANEL_LEFT_EDGE_POS = MARGIN;
VIEWER_PANEL_BOTTOM_EDGE_POS = MARGIN_STATUSBAR; % MARGIN;


%% initialize global variables

nrows = [];
ncols = [];
idxn_current = [];
polygons = [];
labels = [];
index = [];
ROI = table(cell(0,0), zeros(0,0), cell(0,0), cell(0,0), 'VariableNames', {'handle', 'label', 'xy', 'mask'});


%% add the GUI components

warning off;
hs = addUIComponents();
plotImage();

    function hs = addUIComponents
        
        
        %% main figure handle
        hs.MainFigure = figure(...
            'MenuBar', 'none', ...
            'Toolbar', 'figure', ...
            'units', 'pixels', ... %'pixels' 'normalized'
            'Position', [WINDOW_LEFT_EDGE_POS, WINDOW_BOTTOM_EDGE_POS, WINDOW_WIDTH, WINDOW_HEIGHT], ... % [left, bottom, width, height]
            'Name', 'ROI tracer', ...
            'NumberTitle', 'off', ...
            'Resize', 'on', ...
            'CloseRequestFcn', @hCloseMenuCallback, ...
            'SizeChangedFcn', @hSizeChangeCallback, ....
            'WindowButtonDownFcn', @hMouseButtonDownCallback);
        
        
        %% user interface panels
        
        % image panel (left)
        hs.panel.Image = uipanel(...
            'Parent', hs.MainFigure, ...
            'Units', 'pixels', ... %'pixels' 'normalized'
            'Title', '', ...
            'TitlePosition', 'centertop', ...
            'FontSize', 12, ...
            'BackgroundColor', [0, 0, 0], ...
            'BorderType', 'etchedin', ...
            'Position', [VIEWER_PANEL_LEFT_EDGE_POS, VIEWER_PANEL_BOTTOM_EDGE_POS, VIEWER_PANEL_WIDTH, VIEWER_PANEL_HEIGHT]); % [left, bottom, width, height]
        
        % controls/overview panel (right)
        hs.panel.Controls = uipanel(...
            'Parent', hs.MainFigure, ...
            'Units', 'pixels', ... %'pixels' 'normalized'
            'Title', '', ...
            'TitlePosition', 'centertop', ...
            'FontSize', 12, ...
            'BorderType', 'none', ...
            'Position', [CONTROL_PANEL_LEFT_EDGE_POS, CONTROL_PANEL_BOTTOM_EDGE_POS, CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT]);
        
        
        %% user interface controls
        
        % ROI properties -> Label
        hs.controls.LabelText = uicontrol(...
            'Parent', hs.panel.Controls, ...
            'Units', 'normalized',...
            'HandleVisibility', 'callback', ...
            'Position', [0.05, 0.33, 0.9, 0.08], ... % [left, bottom, width, height]
            'Tag', '', ...
            'String', 'Class label (positive integers only)', ...
            'HorizontalAlignment', 'left', ...
            'Style', 'text', ...
            'Enable', 'on');
        
        hs.controls.Label = uicontrol(...
            'Parent', hs.panel.Controls, ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.29, 0.9, 0.08], ... % [left, bottom, width, height]
            'HandleVisibility', 'callback', ...
            'String', {0}, ...
            'Tag', 'Label', ...
            'Value', 0, ...
            'Style', 'edit', ...
            'ToolTipString', 'Assign a class label (positive integer, 0 = unlabelled) to the current ROI polygon', ...
            'FontSize', 14, ...
            'Callback', @(obj, evt) hLabelROI(obj), ...
            'Enable', 'off');
        
        % ROI properties -> New ROI polygon
        hs.controls.NewROIButton = uicontrol(...
            'Parent', hs.panel.Controls, ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.17, 0.9, 0.1], ... % [left, bottom, width, height]
            'HandleVisibility', 'callback', ...
            'String', 'New ROI', ...
            'Tag', 'New', ...
            'Value', false, ...
            'Style', 'pushbutton', ...
            'ToolTipString', 'Add a new ROI polygon', ...
            'Callback', @(obj, evt) hNewROI, ...
            'Enable', 'on');
        
        % ROI properties -> Delete the current ROI polygon
        hs.controls.DeleteROIButton = uicontrol(...
            'Parent', hs.panel.Controls, ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.05, 0.9, 0.1], ... % [left, bottom, width, height]
            'HandleVisibility', 'callback', ...
            'String', 'Delete ROI', ...
            'Tag', 'Delete', ...
            'Value', false, ...
            'Style', 'pushbutton', ...
            'ToolTipString', 'Delete the current ROI polygon', ...
            'Callback', @(obj, evt) hDeleteROI, ...
            'Enable', 'off');
        

        %% toolbar menu
        
        hUipushtool = findall(hs.MainFigure, 'Type', 'uipushtool');
        set(hUipushtool, 'Visible', 'Off')
        set(hUipushtool, 'Separator', 'Off');
        
        hUitoggletool = findall(hs.MainFigure, 'Type', 'uitoggletool');
        set(hUitoggletool, 'Visible', 'Off')
        set(hUitoggletool, 'Separator', 'Off');
        
        hUitogglesplittool = findall(hs.MainFigure, 'Type', 'uitogglesplittool');
        set(hUitogglesplittool, 'Visible', 'Off');
        set(hUitogglesplittool, 'Separator', 'Off');
        
        hPan = findobj(hUitoggletool, 'ToolTipString', 'Pan');
        set(hPan, 'Visible', 'On');
        
        hZoomin = findobj(hUitoggletool, 'ToolTipString', 'Zoom In');
        set(hZoomin, 'Visible', 'On');
        
        hZoomout = findobj(hUitoggletool, 'ToolTipString', 'Zoom Out');
        set(hZoomout, 'Visible', 'On');
        
        hDataCursor = findobj(hUitoggletool, 'ToolTipString', 'Data Cursor');
        set(hDataCursor, 'Visible', 'On');
   
    end


%% callback functions


% resize window callback
    function hSizeChangeCallback(hObject, eventdata)
        
        % Get window geometry
        WINDOW_WIDTH = hObject.Position(3);
        WINDOW_HEIGHT = hObject.Position(4);
        
        if WINDOW_WIDTH > MIN_WINDOW_WIDTH
            
            CONTROL_PANEL_LEFT_EDGE_POS = WINDOW_WIDTH - CONTROL_PANEL_WIDTH - MARGIN;
            VIEWER_PANEL_WIDTH = WINDOW_WIDTH - CONTROL_PANEL_WIDTH - 3 * MARGIN;
            
        end
        
        if WINDOW_HEIGHT > MIN_WINDOW_HEIGHT
            
            CONTROL_PANEL_HEIGHT =  WINDOW_HEIGHT - (MARGIN + MARGIN_STATUSBAR); % WINDOW_HEIGHT - 2 * MARGIN;
            VIEWER_PANEL_HEIGHT =  WINDOW_HEIGHT - (MARGIN + MARGIN_STATUSBAR); % WINDOW_HEIGHT - 2 * MARGIN;
            
        end
        
        % update control panel geometry
        hs.panel.Controls.Position = [CONTROL_PANEL_LEFT_EDGE_POS, CONTROL_PANEL_BOTTOM_EDGE_POS, CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT];
        
        % update image panel geometry
        hs.panel.Image.Position = [VIEWER_PANEL_LEFT_EDGE_POS, VIEWER_PANEL_BOTTOM_EDGE_POS, VIEWER_PANEL_WIDTH, VIEWER_PANEL_HEIGHT];
        
    end

% close window callback
    function hCloseMenuCallback(hObject, eventdata)
        
        % save
        hSaveROI();
        
        selection = ...
            questdlg('Do you really want to exit? Don''t worry, all ROI are saved to workspace :o)', ...
            'Exit', ...
            'Yes', 'No', 'Yes');
        
        if strcmp(selection, 'No')
            
            return;
            
        end
        
        % delete figure handle
        delete(hs.MainFigure); % delete(gcf)
        
    end


% plot image
    function plotImage()
        
        [nrows, ncols, nbands] = size(ima);

        hs.axes.image = axes(...
            'Parent', hs.panel.Image, ...
            'Projection', 'orthographic', ...
            'Units', 'normalized', ...
            'Box', 'on', ...
            'PickableParts', 'all', ... % visible
            'Color', 'none', ... % [0, 0, 0]
            'HitTest', 'on', ...
            'Tag', 'AxisImage');
        
        hs.plot.image = image(ima, ...
            'Parent', hs.axes.image);
        
        axis equal tight
        
    end

    function hMouseButtonDownCallback(obj, evt)
        
        hCheckROI()
        
        if ismember(class(gco), {'matlab.graphics.primitive.Patch'})
            
            disp('mouse button down')
            
            current_poly = get(gco, 'Vertices');
            idxl_current = false(height(ROI),1);
            
            for j = 1:height(ROI)
                
                if isequal(current_poly, getPosition(ROI.handle{j,1}));
                    
                    idxl_current(j,1) = true;
                    
                else
                    
                    idxl_current(j,1) = false;
                    
                end
                
            end
            
            idxn_current = find(idxl_current);
            hSelectROI(idxn_current)

        end
        
    end

% select ROI polygon
    function hSelectROI(idxn)
        
        for j = 1:height(ROI)
            
            API = iptgetapi(ROI.handle{j,1});
            
            if isequal(j, idxn);
                
                API.setColor([1,1,0]);
                
            else
                
                API.setColor([0.2824, 0.2824, 0.9725]);
                
            end
            
        end
        
        hs.controls.Label.String = {ROI.label(idxn,1)};
        hs.controls.Label.Value = ROI.label(idxn,1);
        
    end

% assign class label to current ROI polygon
    function hLabelROI(obj)
        
        
        hCheckROI();
        
        idString = regexp(obj.String{:}, '^\d+$', 'match');
        
        % check if input string is an integer
        switch ~isempty(idString)
            
            case true % string is integer

                ROI.label(idxn_current) = str2double(idString);
                obj.ForegroundColor = [0, 0, 0];
                
            case false % string is not integer
                
                obj.ForegroundColor = [1, 0, 0];
                beep
                
        end
        
    end

% add new ROI polygon
    function hNewROI(hObject, eventdata)
        
        hCheckROI();

        k = height(ROI) + 1;
        idxn_current = k;
        ROI.handle{k,1} = impoly();
        fcn = makeConstrainToRectFcn('impoly', [0.5, size(ima,2)+0.5], [0.5, size(ima,1)+0.5]); % hs.axes.image.XLim, hs.axes.image.YLim
        setPositionConstraintFcn(ROI.handle{k,1}, fcn);
        addNewPositionCallback(ROI.handle{k,1}, @(obj, evt) alert);
        
        ROI.label(k,1) = 0;
        
        % a = findall(hs.MainFigure, 'Type', 'Patch')
        % findall(hs.MainFigure, 'Type', 'uitoggletool');
        
        hSelectROI(height(ROI))

        if height(ROI) >= 1
            
            hs.controls.DeleteROIButton.Enable = 'on';
            hs.controls.Label.Enable = 'on';
            
        end
        
    end

% delete current ROI polygon
    function hDeleteROI(hObject, eventdata)
        
        if height(ROI) >= 1
            
            delete(ROI.handle{idxn_current,1});
            ROI(idxn_current,:) = [];
            
            hCheckROI();
            
        end
        
        if height(ROI) >= 1
            
            hs.controls.DeleteROIButton.Enable = 'on';
            hs.controls.Label.Enable = 'on';
            idxn_current = height(ROI);
            hSelectROI(idxn_current)
            
        else
            
            hs.controls.DeleteROIButton.Enable = 'off';
            hs.controls.Label.Enable = 'off';
            
        end
        
    end

% check ROI
    function hCheckROI()

        idxl_valid = cellfun(@isvalid, ROI.handle, 'UniformOutput', true);
        ROI(~idxl_valid,:) = [];

    end

    function alert()
        
        % disp('polygon edited')
        
    end

% save to workspace
    function hSaveROI()
       
        n_ROI = height(ROI);
        polygons = cell(n_ROI,1);
        index = cell(n_ROI,1);
        labels = ROI.label;
 
        for j = 1:n_ROI
            
            polygons{j,1} = getPosition(ROI.handle{j,1});
            ROI.mask{j,1} = poly2mask(polygons{j,1}(:,1), polygons{j,1}(:,2), nrows, ncols);
            index{j,1} = find(ROI.mask{j,1});
            
        end
        
        % assignin('base', 'ROI', ROI);
        assignin('base', 'polygons', polygons);
        assignin('base', 'index', index);
        assignin('base', 'labels', labels);
        
        n_unlabelled = nnz(ROI.label == 0);
        
        if n_unlabelled > 0
            
            hWarning = warndlg(sprintf('%u unlabelled (label = 0) polygon(s)', n_unlabelled), 'Warning!');
            uiwait(hWarning)
            
        end

    end

end