%% 1. RAG类不同问题的指标分析图
% Contextual Precision 
% 创建示例数据
clc;clear;
data1 = [0.92556, 0.82643; 0.97258, 0.98172; 0.920586, 0.84917]; % 三行两列数据，每行代表一个例子，每列代表一种图例

% 设置X轴位置
x_positions = [1, 2, 3]; % 三个例子的X轴位置

% 创建图形
figure('Position', [100, 100, 400, 300]); % 设置图形位置和大小

% 绘制折线图
hold on; % 允许在同一图形上绘制多个对象

% 绘制第一种图例的数据点和连接线
plot(x_positions, data1(:,1), 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.1,0.7,0.7], 'Color', [0.1,0.7,0.7], 'DisplayName', '查询类');

% 绘制第二种图例的数据点和连接线
plot(x_positions, data1(:,2), 's-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.3, 0.5, 0.9], 'Color', [0.3, 0.5, 0.9], 'DisplayName', '应用类');

% 在每个数据点上显示数值
% 计算数据点之间的垂直距离，用于确定标签偏移量
min_distance = min(abs(data1(:,1) - data1(:,2)))+0.03;
offset = min_distance / 2; % 使用最小距离的一半作为偏移量

% 在每个数据点上显示数值
for i = 1:length(x_positions)
    % 判断两个点的上下关系，智能放置标签
    if data1(i,1) > data1(i,2)
        % 查询类点在上，应用类点在下
        text(x_positions(i), data1(i,1) + offset, num2str(data1(i,1), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', [0.1,0.7,0.7], ...
            'FontWeight', 'bold');
        
        text(x_positions(i), data1(i,2) - offset, num2str(data1(i,2), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'Color', [0.3, 0.5, 0.9], ...
            'FontWeight', 'bold');
    else
        % 应用类点在上，查询类点在下
        text(x_positions(i), data1(i,1) - offset, num2str(data1(i,1), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'Color', [0.1,0.7,0.7], ...
            'FontWeight', 'bold');
        
        text(x_positions(i), data1(i,2) + offset, num2str(data1(i,2), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', [0.3, 0.5, 0.9], ...
            'FontWeight', 'bold');
    end
end

hold off; % 关闭图形叠加模式

% 设置X轴标签
x_labels = {'RAG', 'Self-RAG', 'LightRAG'};
set(gca, 'XTick', 1:3, 'XTickLabel', x_labels, 'FontSize', 12,'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Contextual Precision', 'FontSize', 14,'FontWeight', 'bold');
ylim([0.7, 1.1]); 

% 添加图例
legend('查询类', '应用类', 'Location', 'northeast', 'FontSize', 12,'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white', 'Box', 'off');

%% 2. RAG类不同问题的指标分析图
% Contextual Relevancy 
% 创建示例数据
clc;clear;
data2 = [0.49914, 0.45683; 0.52262, 0.49743; 0.47624, 0.42853]; % 三行两列数据，每行代表一个例子，每列代表一种图例

% 设置X轴位置
x_positions = [1, 2, 3]; % 三个例子的X轴位置

% 创建图形
figure('Position', [100, 100, 400, 300]); % 设置图形位置和大小

% 绘制折线图
hold on; % 允许在同一图形上绘制多个对象

% 绘制第一种图例的数据点和连接线
plot(x_positions, data2(:,1), 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.1,0.7,0.7], 'Color', [0.1,0.7,0.7], 'DisplayName', '查询类');

% 绘制第二种图例的数据点和连接线
plot(x_positions, data2(:,2), 's-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.3, 0.5, 0.9], 'Color', [0.3, 0.5, 0.9], 'DisplayName', '应用类');

% 计算数据点之间的垂直距离，用于确定标签偏移量
min_distance = min(abs(data2(:,1) - data2(:,2)));
offset = min_distance / 2; % 使用最小距离的一半作为偏移量

% 在每个数据点上显示数值
for i = 1:length(x_positions)
    % 判断两个点的上下关系，智能放置标签
    if data2(i,1) > data2(i,2)
        % 查询类点在上，应用类点在下
        text(x_positions(i), data2(i,1) + offset, num2str(data2(i,1), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', [0.1,0.7,0.7], ...
            'FontWeight', 'bold');
        
        text(x_positions(i), data2(i,2) - offset, num2str(data2(i,2), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'Color', [0.3, 0.5, 0.9], ...
            'FontWeight', 'bold');
    else
        % 应用类点在上，查询类点在下
        text(x_positions(i), data2(i,1) - offset, num2str(data2(i,1), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'Color', [0.1,0.7,0.7], ...
            'FontWeight', 'bold');
        
        text(x_positions(i), data2(i,2) + offset, num2str(data2(i,2), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', [0.3, 0.5, 0.9], ...
            'FontWeight', 'bold');
    end
end

hold off; % 关闭图形叠加模式

% 设置X轴标签
x_labels = {'RAG', 'Self-RAG', 'LightRAG'};
set(gca, 'XTick', 1:3, 'XTickLabel', x_labels, 'FontSize', 12,'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Contextual Relevancy', 'FontSize', 14,'FontWeight', 'bold');
ylim([0.4, 0.6]); 

% 添加图例
legend('查询类', '应用类', 'Location', 'northeast', 'FontSize', 12,'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white', 'Box', 'off');
%% 3. RAG类不同问题的指标分析图
% Answer Relevancy
% 创建示例数据
clc;clear;
data1 = [0.61600, 0.61349; 0.54892, 0.55505; 0.56389, 0.58570]; % 三行两列数据，每行代表一个例子，每列代表一种图例

% 设置X轴位置
x_positions = [1, 2, 3]; % 三个例子的X轴位置

% 创建图形
figure('Position', [100, 100, 400, 300]); % 设置图形位置和大小

% 绘制折线图
hold on; % 允许在同一图形上绘制多个对象

% 绘制第一种图例的数据点和连接线
plot(x_positions, data1(:,1), 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.1,0.7,0.7], 'Color', [0.1,0.7,0.7], 'DisplayName', '查询类');

% 绘制第二种图例的数据点和连接线
plot(x_positions, data1(:,2), 's-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.3, 0.5, 0.9], 'Color', [0.3, 0.5, 0.9], 'DisplayName', '应用类');

% 在每个数据点上显示数值
% 计算数据点之间的垂直距离，用于确定标签偏移量
min_distance = min(abs(data1(:,1) - data1(:,2)))+0.03;
offset = min_distance / 2; % 使用最小距离的一半作为偏移量

% 在每个数据点上显示数值
for i = 1:length(x_positions)
    % 判断两个点的上下关系，智能放置标签
    if data1(i,1) > data1(i,2)
        % 查询类点在上，应用类点在下
        text(x_positions(i), data1(i,1) + offset, num2str(data1(i,1), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', [0.1,0.7,0.7], ...
            'FontWeight', 'bold');
        
        text(x_positions(i), data1(i,2) - offset, num2str(data1(i,2), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'Color', [0.3, 0.5, 0.9], ...
            'FontWeight', 'bold');
    else
        % 应用类点在上，查询类点在下
        text(x_positions(i), data1(i,1) - offset, num2str(data1(i,1), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'Color', [0.1,0.7,0.7], ...
            'FontWeight', 'bold');
        
        text(x_positions(i), data1(i,2) + offset, num2str(data1(i,2), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', [0.3, 0.5, 0.9], ...
            'FontWeight', 'bold');
    end
end

hold off; % 关闭图形叠加模式

% 设置X轴标签
x_labels = {'RAG', 'Self-RAG', 'LightRAG'};
set(gca, 'XTick', 1:3, 'XTickLabel', x_labels, 'FontSize', 12,'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Answer Relevancy', 'FontSize', 14,'FontWeight', 'bold');
ylim([0.4, 0.8]); 

% 添加图例
legend('查询类', '应用类', 'Location', 'northeast', 'FontSize', 12,'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white', 'Box', 'off');

%% 2. RAG类不同问题的指标分析图
% Faithfulness
% 创建示例数据
clc;clear;
%data2 = [0.58501, 0.61775; 0.59971, 0.60544; 0.59992, 0.65218]; % 三行两列数据，每行代表一个例子，每列代表一种图例
data2 = [0.58501, 0.61775; 0.59971, 0.60544; 0.59992, 0.65218];
% 设置X轴位置
x_positions = [1, 2, 3]; % 三个例子的X轴位置

% 创建图形
figure('Position', [100, 100, 400, 300]); % 设置图形位置和大小

% 绘制折线图
hold on; % 允许在同一图形上绘制多个对象

% 绘制第一种图例的数据点和连接线
plot(x_positions, data2(:,1), 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.1,0.7,0.7], 'Color', [0.1,0.7,0.7], 'DisplayName', '查询类');

% 绘制第二种图例的数据点和连接线
plot(x_positions, data2(:,2), 's-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.3, 0.5, 0.9], 'Color', [0.3, 0.5, 0.9], 'DisplayName', '应用类');

% 在每个数据点上显示数值
% 计算数据点之间的垂直距离，用于确定标签偏移量
min_distance = min(abs(data2(:,1) - data2(:,2)))+0.03;
offset = min_distance / 2; % 使用最小距离的一半作为偏移量

% 在每个数据点上显示数值
for i = 1:length(x_positions)
    % 判断两个点的上下关系，智能放置标签
    if data2(i,1) > data2(i,2)
        % 查询类点在上，应用类点在下
        text(x_positions(i), data2(i,1) + offset, num2str(data2(i,1), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', [0.1,0.7,0.7], ...
            'FontWeight', 'bold');
        
        text(x_positions(i), data2(i,2) - offset, num2str(data2(i,2), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'Color', [0.3, 0.5, 0.9], ...
            'FontWeight', 'bold');
    else
        % 应用类点在上，查询类点在下
        text(x_positions(i), data2(i,1) - offset, num2str(data2(i,1), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'Color', [0.1,0.7,0.7], ...
            'FontWeight', 'bold');
        
        text(x_positions(i), data2(i,2) + offset, num2str(data2(i,2), '%.2f'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', [0.3, 0.5, 0.9], ...
            'FontWeight', 'bold');
    end
end
hold off; % 关闭图形叠加模式

% 设置X轴标签
x_labels = {'RAG', 'Self-RAG', 'LightRAG'};
set(gca, 'XTick', 1:3, 'XTickLabel', x_labels, 'FontSize', 12,'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Faithfulness', 'FontSize', 14,'FontWeight', 'bold');
ylim([0.4, 0.8]); 

% 添加图例
legend('查询类', '应用类', 'Location', 'northeast', 'FontSize', 12,'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white', 'Box', 'off');













