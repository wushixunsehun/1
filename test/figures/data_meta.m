%% 1. Metagpt Professionalism分析 查询类和应用类对比
clc;clear;
data1 = [0.92917, 0.84969; 0.88750, 0.79785; 0.81417, 0.80694]; % 三行两列数据，每行代表一个例子，每列代表一种图例

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
x_labels = {'qweno', 'dbo', 'sumo'};
set(gca, 'XTick', 1:3, 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_positions, 'XTickLabel', x_labels, 'FontSize', 12, 'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Professionalism', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0.6, 1.1]); % 设置Y轴范围为0-100

% 添加图例
legend('show', 'Location', 'northeast', 'FontSize', 12, 'FontWeight', 'bold');

% 添加标题
%title('三个示例的双组数据点折线图比较', 'FontSize', 16, 'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gca, 'Color', 'white', 'Box', 'off');
set(gcf, 'Color', 'white');

%% 2. Metagpt角色遵守分析 查询类和应用类对比
clc;clear;
data1 = [0.38333, 0.59271; 0.67500, 0.61512; 0.16667, 0.14299]; % 三行两列数据，每行代表一个例子，每列代表一种图例

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
x_labels = {'qweno', 'dbo', 'sumo'};
set(gca, 'XTick', 1:3, 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_positions, 'XTickLabel', x_labels, 'FontSize', 12, 'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Role Adherence', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, 1]); % 设置Y轴范围为0-100

% 添加图例
legend('show', 'Location', 'northeast', 'FontSize', 12, 'FontWeight', 'bold');

% 添加标题
%title('三个示例的双组数据点折线图比较', 'FontSize', 16, 'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gca, 'Color', 'white', 'Box', 'off');
set(gcf, 'Color', 'white');
%% 3. Metagpt Conversation Relevancy分析 查询类和应用类对比
clc;clear;
data1 = [1, 0.98907; 0.98333, 0.91257; 0.99167, 1]; % 三行两列数据，每行代表一个例子，每列代表一种图例

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
x_labels = {'qweno', 'dbo', 'sumo'};
set(gca, 'XTick', 1:3, 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_positions, 'XTickLabel', x_labels, 'FontSize', 12, 'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Conversation Relevancy', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0.8, 1.2]); % 设置Y轴范围为0-100

% 添加图例
legend('show', 'Location', 'northeast', 'FontSize', 12, 'FontWeight', 'bold');

% 添加标题
%title('三个示例的双组数据点折线图比较', 'FontSize', 16, 'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gca, 'Color', 'white', 'Box', 'off');
set(gcf, 'Color', 'white');
%% 4. Metagpt Conversation Completeness分析 查询类和应用类对比
clc;clear;
data1 = [0.88750, 0.79785; 0.81417, 0.80694; 0.98611, 0.97658]; % 三行两列数据，每行代表一个例子，每列代表一种图例

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
x_labels = {'qweno', 'dbo', 'sumo'};
set(gca, 'XTick', 1:3, 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_positions, 'XTickLabel', x_labels, 'FontSize', 12, 'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Conversation Completeness', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0.6, 1.2]); % 设置Y轴范围为0-100

% 添加图例
legend('show', 'Location', 'northeast', 'FontSize', 12, 'FontWeight', 'bold');

% 添加标题
%title('三个示例的双组数据点折线图比较', 'FontSize', 16, 'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gca, 'Color', 'white', 'Box', 'off');
set(gcf, 'Color', 'white');
