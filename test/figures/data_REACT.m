%% 1. REACT类不同问题的指标分析图
% time 
% 创建示例数据
clc;clear;
data1 = [0.50833, 0.35228; 1, 0.94499; 0.78375, 0.73347;0.81667,0.90018;0.57071,0.57004]; % 三行两列数据，每行代表一个例子，每列代表一种图例

% 设置X轴位置
x_positions = [1, 2, 3, 4, 5]; % 三个例子的X轴位置

% 创建图形
figure('Position', [100, 100, 900, 400]); % 设置图形位置和大小
%figure
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
x_labels = {'Role Adherence', 'Conversation Relevancy', 'Task Completion','Tool Correctness','Hallucination'};
set(gca, 'XTick', 1:5, 'XTickLabel', x_labels, 'FontSize', 8,'FontWeight', 'bold');

% 设置Y轴标签和范围
ylabel('Score', 'FontSize', 14,'FontWeight', 'bold');
ylim([0.2, 1.2]); 

% 添加图例
legend('查询类', '应用类', 'Location', 'northeast', 'FontSize', 12,'FontWeight', 'bold');

% 添加网格线
grid on;

% 设置图形背景和边框
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white', 'Box', 'off');
