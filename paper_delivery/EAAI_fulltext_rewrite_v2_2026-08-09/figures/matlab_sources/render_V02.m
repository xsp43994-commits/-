function render_V02(outputRoot)
% 固定真实DSM任务路线图。关键尺寸、颜色和视角均在本文件顶部集中设置。
if nargin < 1
    outputRoot = fileparts(fileparts(mfilename('fullpath')));
end
dataRoot = fullfile(outputRoot, 'source_data', 'V02');
outRoot = fullfile(outputRoot, 'showcase');
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

terrainT = readtable(fullfile(dataRoot, 'terrain.csv'));
roads = readtable(fullfile(dataRoot, 'roads.csv'));
points = readtable(fullfile(dataRoot, 'points.csv'));
routes = readtable(fullfile(dataRoot, 'routes.csv'));

nx = max(terrainT.x) + 1; ny = max(terrainT.y) + 1;
Z = reshape(terrainT.z, [nx, ny])';
nodes = points(strcmp(points.point_type, 'inspection'), :);
airport = points(strcmp(points.point_type, 'airport'), :);
margin = 20;
xlimTask = [max(0, floor(min(nodes.x)-margin)), min(nx-1, ceil(max(nodes.x)+margin))];
ylimTask = [max(0, floor(min(nodes.y)-margin)), min(ny-1, ceil(max(nodes.y)+margin))];
[X, Y] = meshgrid(0:nx-1, 0:ny-1);
maskX = X(1,:) >= xlimTask(1) & X(1,:) <= xlimTask(2);
maskY = Y(:,1) >= ylimTask(1) & Y(:,1) <= ylimTask(2);

fig = figure('Color','w','Units','centimeters','Position',[2 2 17.8 15.0], 'Renderer','opengl');
ax = axes(fig); hold(ax,'on');
surf(ax, X(maskY,maskX), Y(maskY,maskX), Z(maskY,maskX), Z(maskY,maskX), ...
    'EdgeColor','none','FaceAlpha',0.96);
colormap(ax, parula(256));
contour3(ax, X(maskY,maskX), Y(maskY,maskX), Z(maskY,maskX)+1.5, 12, ...
    'LineColor',[0.42 0.42 0.42],'LineWidth',0.45);

roadIds = unique(roads.road_id)';
for rid = roadIds
    r = roads(roads.road_id==rid,:);
    keep = r.x>=xlimTask(1) & r.x<=xlimTask(2) & r.y>=ylimTask(1) & r.y<=ylimTask(2);
    r = r(keep,:);
    if height(r)>1
        plot3(ax,r.x,r.y,r.z+3.0,'Color',[0.25 0.25 0.25],'LineWidth',1.25);
    end
end

priorityColors = [0.30 0.55 0.85; 0.93 0.66 0.20; 0.78 0.20 0.18];
prioritySizes = [28 40 56];
priorityHandles = gobjects(3,1);
for p = 1:3
    q = nodes(nodes.priority==p,:);
    priorityHandles(p) = scatter3(ax,q.x,q.y,q.z+7,prioritySizes(p),priorityColors(p,:), ...
        'filled','MarkerEdgeColor','w','LineWidth',0.75);
end
airportHandle = scatter3(ax,airport.x,airport.y,airport.z+10,110,'p','filled', ...
    'MarkerFaceColor',[0.05 0.05 0.05],'MarkerEdgeColor','w','LineWidth',1.0);

models = {'full','a2c_pointer','traditional_ppo','milp'};
modelLabels = {'PPO+Pointer','A2C+Pointer','传统PPO','MILP'};
modelColors = [35 105 189; 230 134 25; 42 157 143; 34 34 34]/255;
lineStyles = {'-','--','-.',':'};
routeHandles = gobjects(4,1);
for i = 1:numel(models)
    r = routes(strcmp(routes.model,models{i}) & strcmp(routes.status,'route'),:);
    r = sortrows(r,'sequence');
    routeHandles(i) = plot3(ax,r.x,r.y,r.z+12, 'Color',modelColors(i,:), ...
        'LineStyle',lineStyles{i},'LineWidth',2.15);
end

axis(ax,'tight'); axis(ax,'vis3d');
xlim(ax,xlimTask); ylim(ax,ylimTask);
view(ax,-38,36); camproj(ax,'perspective'); camzoom(ax,0.80);
grid(ax,'off'); box(ax,'on');
xlabel(ax,'Local Easting (30 m/grid)','FontName','Arial');
ylabel(ax,'Local Northing (30 m/grid)','FontName','Arial');
zlabel(ax,'Elevation (m)','FontName','Arial');
set(ax,'FontName','Arial','FontSize',8,'LineWidth',0.8,'TickDir','out');
cb = colorbar(ax); cb.Label.String = 'Terrain elevation (m)'; cb.Label.FontName = 'Arial';
lgd = legend([routeHandles; airportHandle; priorityHandles], ...
    [modelLabels, {'机场','低优先级','中优先级','高优先级'}], ...
    'Location','southoutside','NumColumns',4,'Box','off','FontName','Microsoft YaHei','FontSize',7);
% 为三维坐标框、色标和图例分配互不重叠的固定区域。
set(ax,'Position',[0.055 0.290 0.705 0.600]);
set(cb,'Position',[0.845 0.305 0.026 0.565]);
set(lgd,'Position',[0.115 0.018 0.710 0.105]);

stem = fullfile(outRoot,'V02_固定真实DSM地形路线');
% 固定页面尺寸，避免exportgraphics自动紧裁切导致三维坐标框贴边。
set(fig,'PaperUnits','centimeters','PaperPosition',[0 0 17.8 15.0], ...
    'PaperSize',[17.8 15.0],'PaperPositionMode','manual','InvertHardcopy','off');
print(fig,[stem '.pdf'],'-dpdf','-painters');
print(fig,[stem '.png'],'-dpng','-r600');
print(fig,[stem '.tiff'],'-dtiff','-r600');
print(fig,[stem '.svg'],'-dsvg','-painters');
savefig(fig,[stem '.fig']);
close(fig);
end
