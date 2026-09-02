# 强化学习路径规划论文图目惯例审计

## 审计目的

本审计只用于决定保研面试PPT的图目组合，不改变任何项目结果。判断标准不是简单复制权威论文，而是识别该领域读者熟悉的证据结构，再用本项目更严格的统计图实现相同证据角色。

## 原始研究中的高频图目

1. **训练回报或成功率曲线**
   - 上传的 Tang、Liang 与 Li（*Drones*, 2024）Figure 5 展示三种算法的累计回报训练曲线，Figure 12 同时展示动态场景中的回报与完成率。
   - Hodge、Hawkins 与 Alexander（*Neural Computing and Applications*, 2021）Figure 7 展示多种PPO配置的平均训练回报，Figure 8展示训练回报标准差。
   - 多无人机和PPO路径规划研究也常用平均回报、成功率、步数随训练迭代变化的曲线。

2. **路线或轨迹可视化**
   - 上传论文Figure 7比较多算法规划路线，Figure 13展示动态障碍条件下的连续轨迹快照。
   - 该图型用于说明规划行为和空间合理性，但通常属于描述性场景证据，不能代替跨任务统计。

3. **最终量化比较**
   - 上传论文Figure 8比较路径长度、规划时间和转折点。
   - 同领域论文常报告平均回报、成功率、路径长度、运行时间或步数；本项目的M01以地图级分布代替均值柱图，统计层级更严格。

4. **机制、消融或参数敏感性**
   - 并非所有路径规划论文都完成严格消融，但这类图能回答“性能来自哪里”，对科研面试价值很高。
   - 本项目M10采用地图级配对效应、bootstrap区间和Holm校正，证据强度高于只比较消融终值的均值图。

## 对本项目图目的决定

| 证据角色 | 主图 | 决策 |
|---|---|---|
| 空间行为与应用场景 | V02 | 保留；与领域常见路线图一致，但明确为真实DSM仿真而非实飞 |
| 最终跨地图效果 | M01 | 保留；虽然半眼分布不是该领域最常见模板，但比均值柱图更能保留地图级变异 |
| 强化学习训练过程 | S09 | 升为主图；最直接对应领域常见的平均/累计回报曲线，同时保留五种子与IQR且不做时间平滑 |
| 机制证据 | M10 | 保留；训练曲线无法替代地图级配对消融统计 |
| 七模型训练动态 | S06 | 作为备选；视觉信息丰富，但证据与S09重叠，且训练批次表现不能替代外部泛化 |
| 固定外部验证动态 | M06 | 作为备选；科学语义严谨，在被追问回报与外部性能是否一致时使用 |

## 主要原始文献

- Tang, J., Liang, Y., & Li, K. (2024). Dynamic Scene Path Planning of UAVs Based on Deep Reinforcement Learning. *Drones*, 8, 60. DOI: https://doi.org/10.3390/drones8020060
- Hodge, V. J., Hawkins, R., & Alexander, R. (2021). Deep reinforcement learning for drone navigation using sensor data. *Neural Computing and Applications*, 33, 2015-2033. https://link.springer.com/article/10.1007/s00521-020-05097-x
- Puente-Castro, A. et al. (2022). UAV swarm path planning with reinforcement learning for field prospecting. *Applied Intelligence*, 52, 14101-14118. https://link.springer.com/article/10.1007/s10489-022-03254-4
- Multi-UAV Path Planning in GPS and Communication Denial Environment. *Sensors*, 23, 2997. https://www.mdpi.com/1424-8220/23/6/2997
- Path Planning for Unmanned Surface Vehicles with Strong Generalization Ability Based on Improved Proximal Policy Optimization. *Sensors*, 23, 8864. https://doi.org/10.3390/s23218864
