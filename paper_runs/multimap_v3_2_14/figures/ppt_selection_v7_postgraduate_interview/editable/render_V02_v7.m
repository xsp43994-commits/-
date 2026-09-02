function render_V02_v7(outputRoot)
% v7真实DSM路线图。只读取冻结V02 Source Data；尺寸、颜色和视角集中设置。
if nargin < 1
    outputRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
dataRoot = fullfile(outputRoot, 'source_data', 'V02');

terrainT = readtable(fullfile(dataRoot, 'terrain.csv'));
roads = readtable(fullfile(dataRoot, 'roads.csv'));
points = readtable(fullfile(dataRoot, 'points.csv'));
routes = readtable(fullfile(dataRoot, 'routes.csv'));

nx = max(terrainT.x) + 1; ny = max(terrainT.y) + 1;
Z = reshape(terrainT.z, [nx, ny])';
[X, Y] = meshgrid(0:nx-1, 0:ny-1);
nodes = points(strcmp(points.point_type, 'inspection'), :);
airport = points(strcmp(points.point_type, 'airport'), :);
margin = 20;
xlimTask = [max(0, floor(min(nodes.x)-margin)), min(nx-1, ceil(max(nodes.x)+margin))];
ylimTask = [max(0, floor(min(nodes.y)-margin)), min(ny-1, ceil(max(nodes.y)+margin))];
maskX = X(1,:) >= xlimTask(1) & X(1,:) <= xlimTask(2);
maskY = Y(:,1) >= ylimTask(1) & Y(:,1) <= ylimTask(2);

for language = {'zh','en'}
    lang = language{1};
    outRoot = fullfile(outputRoot, 'exports', lang, 'visual');
    if ~exist(outRoot, 'dir'), mkdir(outRoot); end
    fig = figure('Color','w','Units','centimeters','Position',[2 2 19.0 15.0], 'Renderer','opengl');
    ax = axes(fig); hold(ax,'on');
    surf(ax, X(maskY,maskX), Y(maskY,maskX), Z(maskY,maskX), Z(maskY,maskX), ...
        'EdgeColor','none','FaceAlpha',0.94);
    colormap(ax, parula(256));
    contour3(ax, X(maskY,maskX), Y(maskY,maskX), Z(maskY,maskX)+1.5, 12, ...
        'LineColor',[0.48 0.48 0.48],'LineWidth',0.45);

    roadIds = unique(roads.road_id)';
    for rid = roadIds
        r = roads(roads.road_id==rid,:);
        keep = r.x>=xlimTask(1) & r.x<=xlimTask(2) & r.y>=ylimTask(1) & r.y<=ylimTask(2);
        r = r(keep,:);
        if height(r)>1, plot3(ax,r.x,r.y,r.z+3.0,'Color',[0.30 0.30 0.30],'LineWidth',1.0); end
    end

    priorityColors = [86 180 233; 230 159 0; 213 94 0]/255;
    prioritySizes = [26 39 55]; priorityHandles = gobjects(3,1);
    for p = 1:3
        q = nodes(nodes.priority==p,:);
        priorityHandles(p) = scatter3(ax,q.x,q.y,q.z+7,prioritySizes(p),priorityColors(p,:), ...
            'filled','MarkerEdgeColor','w','LineWidth',0.65);
    end
    airportHandle = scatter3(ax,airport.x,airport.y,airport.z+10,100,'p','filled', ...
        'MarkerFaceColor',[0.05 0.05 0.05],'MarkerEdgeColor','w','LineWidth',0.8);

    models = {'full','a2c_pointer','traditional_ppo','milp'};
    modelColors = [0 114 178; 230 159 0; 0 158 115; 45 45 45]/255;
    lineStyles = {'-','--','-.',':'}; routeHandles = gobjects(4,1);
    for i = 1:numel(models)
        r = routes(strcmp(routes.model,models{i}) & strcmp(routes.status,'route'),:);
        r = sortrows(r,'sequence');
        routeHandles(i) = plot3(ax,r.x,r.y,r.z+12,'Color',modelColors(i,:), ...
            'LineStyle',lineStyles{i},'LineWidth',1.45);
    end

    axis(ax,'tight'); axis(ax,'vis3d'); xlim(ax,xlimTask); ylim(ax,ylimTask);
    view(ax,-38,36); camproj(ax,'perspective'); camzoom(ax,0.84);
    grid(ax,'on'); box(ax,'on');
    set(ax,'GridColor',[0.84 0.86 0.88],'GridAlpha',0.75,'FontName','Arial', ...
        'FontSize',8,'LineWidth',0.8,'TickDir','out');
    if strcmp(lang,'en')
        xlabel(ax,'Local Easting (30 m/grid)'); ylabel(ax,'Local Northing (30 m/grid)'); zlabel(ax,'Elevation (m)');
        cbLabel = 'Terrain elevation (m)'; modelLabels = {'PPO+Pointer','A2C+Pointer','Traditional PPO','MILP'};
        extraLabels = {'Depot','Low priority','Medium priority','High priority'};
    else
        xlabel(ax,'局部东向坐标（30 m/格）','FontName','Microsoft YaHei');
        ylabel(ax,'局部北向坐标（30 m/格）','FontName','Microsoft YaHei');
        zlabel(ax,'高程（m）','FontName','Microsoft YaHei');
        cbLabel = '地形高程（m）'; modelLabels = {'PPO+Pointer','A2C+Pointer','传统PPO','MILP'};
        extraLabels = {'机场','低优先级','中优先级','高优先级'};
    end
    cb = colorbar(ax); cb.Label.String = cbLabel; cb.Label.FontName = 'Microsoft YaHei'; cb.FontName='Arial'; cb.FontSize=8;
    lgd = legend([routeHandles; airportHandle; priorityHandles],[modelLabels,extraLabels], ...
        'Location','southoutside','NumColumns',4,'Box','on','FontName','Microsoft YaHei','FontSize',8);
    set(lgd,'Color','w','EdgeColor',[0.68 0.70 0.73],'LineWidth',0.55);
    % 为横轴标题与底部图例保留明确间隔，避免实际尺寸下文字相互遮挡。
    set(ax,'Position',[0.055 0.340 0.725 0.555]); set(cb,'Position',[0.865 0.340 0.024 0.535]);
    set(lgd,'Position',[0.105 0.010 0.745 0.100]);

    stem = fullfile(outRoot,['V02_v7_drones_style_' lang]);
    set(fig,'PaperUnits','centimeters','PaperPosition',[0 0 19.0 15.0], ...
        'PaperSize',[19.0 15.0],'PaperPositionMode','manual','InvertHardcopy','off');
    print(fig,[stem '.pdf'],'-dpdf','-painters');
    print(fig,[stem '.png'],'-dpng','-r600');
    print(fig,[stem '.tiff'],'-dtiff','-r600');
    savefig(fig,[stem '.fig']);
    close(fig);
end
end
