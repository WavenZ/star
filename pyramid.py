import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def get_distance(u, v, params):
    
    # 将参数解包
    cx, cy, dx, dy, focal = params[:5]

    # 计算 u 的归一化方向矢量
    x, y = (u[0] - cx) * dx, (u[1] - cy) * dy
    norm = np.sqrt(x * x + y * y + focal * focal)
    u1, u2, u3 = [-x, -y, focal] / norm
    
    # 计算 v 的归一化方向矢量
    x, y = (v[0] - cx) * dx, (v[1] - cy) * dy
    norm = np.sqrt(x * x + y * y + focal * focal)
    v1, v2, v3 = [-x, -y, focal] / norm

    # 计算角距
    ans = np.arccos(v1 * u1 + v2 * u2 + v3 * u3) * 180 / np.pi

    return ans

def f_calibration(center, params):
    
    # 将参数解包
    [cx, cy, dx, dy, focal] = params[:5]
    [ep1, ep2, eq1, eq2] = params[5:]

    # 图像坐标系列转换为星敏感器坐标系
    x, y = (center[:, 0] - cx) * dx, (center[:, 1] - cy) * dy
    r2 = x * x + y * y

    DX = (x * ((eq1 * r2) + (eq2 * r2 * r2))) + ((ep1 * (r2 + (2 * x * x))) + (2 * ep2 * x * y))
    DY = (y * ((eq1 * r2) + (eq2 * r2 * r2))) + ((ep2 * (r2 + (2 * y * y))) + (2 * ep1 * x * y))

    xyz = np.array([DX - x, DY - y, focal * np.ones(x.shape[0])])
    xyz = (xyz / np.sqrt(np.sum(xyz * xyz, 0)))

    return xyz

def verify_pyramid(candidate, centers, params, err, pairs_catalog, catalog, group_cnt, group_start):
    '''金字塔验证
    '''

    # 三角形顶点的三个星点
    a, b, c = candidate[:3] - 1

    # 其余的星点
    all = np.zeros(centers.shape[0])
    all[candidate[3: 6]] = 1
    others = np.where(all == 0)[0]

    # 遍历其余所有星点
    for o in others:
        # 计算当前点到三角形三个顶点的角距
        d1 = get_distance(centers[candidate[3], :], centers[o, :], params)
        d2 = get_distance(centers[candidate[4], :], centers[o, :], params)
        d3 = get_distance(centers[candidate[5], :], centers[o, :], params)
        
        # 角距离散化
        gn = np.round(np.array([d1, d2, d3]) / 20 * 1000).astype(np.int32) - 1
        
        # 过滤极限情况角距
        if np.any(gn - (err - 1) <= 0) or np.any(gn + err + 2 >= 1000):
            continue

        # 找到星库中距离 a 角距约为 d1 的星
        pairs = pairs_catalog[group_start[max(gn[0] - (err - 1), 0)]:
                              group_start[min(gn[0] + err + 2, 999)], 1:]
        index = np.where(np.logical_or(pairs[:, 0] == a, pairs[:, 1] == a))[0]
        sa = np.sum(pairs[index, :], 1) - a
    
        # 找到星库中距离 b 角距约为 d2 的星
        pairs = pairs_catalog[group_start[max(gn[1] - (err - 1), 0)]:
                              group_start[min(gn[1] + err + 2, 999)], 1:]
        index = np.where(np.logical_or(pairs[:, 0] == b, pairs[:, 1] == b))[0]
        sb = np.sum(pairs[index, :], 1) - b

        # 找到星库中距离 c 角距约为 d3 的星
        pairs = pairs_catalog[group_start[max(gn[2] - (err - 1), 0)]:
                              group_start[min(gn[2] + err + 2, 999)], 1:]
        index = np.where(np.logical_or(pairs[:, 0] == c, pairs[:, 1] == c))[0]
        sc = np.sum(pairs[index, :], 1) - c

        # 找到满足上述三个角距要求的星
        sabc = np.intersect1d(sa, np.intersect1d(sb, sc))

        # 如果匹配到的星的数量不是 1（匹配失败或者匹配到多个星），则继续匹配
        if sabc.shape[0] != 1:
            continue
        # 否则，返回匹配的结果
        else:
            return np.hstack((candidate[3:], [o, a, b, c], sabc)).astype(np.int32)
    
    return np.array([])

def quest_s_eig(ww_cal, vv_cal):
    
    z = np.zeros((3, 1))
    ndim = ww_cal.shape[0]
    sigma = np.sum(ww_cal * vv_cal) / ndim
    z[0] = (ww_cal[:, 1].T.dot(vv_cal[:, 2]) - ww_cal[:, 2].T.dot(vv_cal[:, 1])) / ndim
    z[1] = (ww_cal[:, 2].T.dot(vv_cal[:, 0]) - ww_cal[:, 0].T.dot(vv_cal[:, 2])) / ndim
    z[2] = (ww_cal[:, 0].T.dot(vv_cal[:, 1]) - ww_cal[:, 1].T.dot(vv_cal[:, 0])) / ndim


    s = (ww_cal.T.dot(vv_cal) + vv_cal.T.dot(ww_cal)) / ndim
    
    K = np.vstack((np.hstack((s - sigma * np.eye(3), z)), np.append(z, sigma)))
    
    lambd, qs = np.linalg.eig(K)
    id = np.argmin(abs(lambd - 1))

    return qs[:, id]

def quest_new(ww_cal, vv_cal):
    qout = quest_s_eig(ww_cal, vv_cal)
    if abs(qout[3]) > 0.5:
        return qout
    if abs(qout[0]) >= abs(qout[1]) and abs(qout[0]) >= abs(qout[2]):
        vv_cal[:, 1:3] = -vv_cal[:, 1:3]
        qout = quest_s_eig(ww_cal, vv_cal)
        qout = np.array([-qout[3], qout[2], -qout[1], qout[0]]).T
        return qout
    if abs(qout[1]) >= abs(qout[2]):
        vv_cal[:, 0] = -vv_cal[:, 0]
        vv_cal[:, 2] = -vv_cal[:, 2]
        qout = quest_s_eig(ww_cal, vv_cal)
        qout = np.array([-qout[2], -qout[3], qout[0], qout[1]]).T
        return qout
    vv_cal[:, :2] = -vv_cal[:, :2]
    qout = quest_s_eig(ww_cal, vv_cal)
    qout = np.array([qout[1], -qout[0], -qout[3], qout[2]]).T
    return qout

def quest(ww_cal, vv_cal):
    
    # 计算四元数
    qout = quest_new(ww_cal, vv_cal)
    
    # 四元数转姿态矩阵
    current_att = np.zeros((3, 3))
    current_att[0, 0] = qout[0] * qout[0] - qout[1] * qout[1] - qout[2] * qout[2] + qout[3] * qout[3]
    current_att[0, 1] = 2 * (qout[0] * qout[1] + qout[2] * qout[3])
    current_att[0, 2] = 2 * (qout[0] * qout[2] - qout[1] * qout[3])
    current_att[1, 0] = 2 * (qout[0] * qout[1] - qout[2] * qout[3])
    current_att[1, 1] = - qout[0] * qout[0] + qout[1] * qout[1] - qout[2] * qout[2] + qout[3] * qout[3]
    current_att[1, 2] = 2 * (qout[1] * qout[2] + qout[0] * qout[3])
    current_att[2, 0] = 2 * (qout[0] * qout[2] + qout[1] * qout[3])
    current_att[2, 1] = 2 * (qout[1] * qout[2] - qout[0] * qout[3])
    current_att[2, 2] = - qout[0] * qout[0] - qout[1] * qout[1] + qout[2] * qout[2] + qout[3] * qout[3]

    # 姿态矩阵转欧拉角
    ra = np.arctan2(current_att[2, 1], current_att[2, 0])
    dec = np.arcsin(current_att[2, 2])
    roll = np.arctan2(current_att[0, 2], current_att[1, 2])

    if current_att[2, 0] < 0:
        ra += np.pi

    if current_att[2, 0] > 0 and current_att[2, 1] < 0:
        ra += 2 * np.pi

    return (np.array([ra, dec, roll]), qout)

def quaternion_to_rotation_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0]],
                           [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0]],
                           [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2]]], dtype=q.dtype)
    return rot_matrix
    
def verification(pairs_catalog, q, ww_cal, th):

    # quat2dcm
    dcm = quaternion_to_rotation_matrix(q[[3, 0, 1, 2]])
    

    vv_cal = ww_cal.T.dot(dcm)
    d = vv_cal.dot(pairs_catalog[:, 1:4].T)
    
    id = np.argmax(d, 1)
    dm = np.max(d, 1)
    dm[np.where(dm > 1)] = 1

    index = np.where(dm > np.cos(th * np.pi / 180))
    star_id = - np.ones(ww_cal.shape[1]).astype(np.int32)
    star_id[index] = id[index]
    # res = np.max(np.arccos(dm[index]))
    return star_id



def pyramid(centers_origin, params, err, match_min, catalog, pairs_catalog, group_start, group_cnt):
    
    # 用来进行星图识别的最多恒星数
    num_use = min(centers_origin.shape[0], 64)

    if num_use < 3:
        return - np.zeros(centers.shape[0])

    centers = centers_origin[:num_use, :]

    # 星表的大小
    size_catalog = catalog.shape[0]

    # 成功识别
    identified = False

    # 候选三角形
    candidates = []
    candidate3s = []

    # 计算两两之间的角距
    dis = np.zeros((num_use, num_use))
    gns = np.zeros((num_use, num_use)).astype(np.int32)

    for i in range(num_use - 1):
        for j in range(i + 1, num_use):
            angle = get_distance(centers[i], centers[j], params)
            dis[i][j] = angle
            gns[i][j] = round(angle / 20 * 1000)

    # 对称化两个矩阵
    gns = gns + gns.T
    dis = dis + dis.T
    for dj in range(num_use - 2):
        for dk in range(num_use - dj - 1):
            for i in range(num_use - dj - dk):
                
                # 以组合的方式选择观测三角形
                j, k = i + dj, i + dj + dk
                gn = np.array([gns[i, j], gns[j, k], gns[k, i]]) - 1

                # 过滤边界条件
                if np.sum(gn - (err - 1) <= 0) or np.sum(gn + err + 2 >= 1000):
                    continue
                
                pair_cnt = np.zeros(size_catalog + 1).astype(np.int32)
                pair_map = np.zeros((size_catalog + 1, 100)).astype(np.int32)
                pair_cnt2 = np.zeros(size_catalog + 1).astype(np.int32)
                pair_map2 = np.zeros((size_catalog + 1, 100, 2)).astype(np.int32)
                pairs = pairs_catalog[group_start[max(gn[0] - (err - 1), 0)]:
                                      group_start[min(gn[0] + err + 2, 999)], 1:]
                for pair in pairs:
                    u, v = pair.astype(np.int32) + 1
                    pair_map[u, pair_cnt[u]] = v
                    pair_cnt[u] += 1
                    pair_map[v, pair_cnt[v]] = u
                    pair_cnt[v] += 1

                pairs = pairs_catalog[group_start[max(gn[1] - (err - 1), 0)]:
                                      group_start[min(gn[1] + err + 2, 999)], 1:]

                
                for pair in pairs:
                    u, v = pair.astype(np.int32) + 1
                    for w in pair_map[u, :pair_cnt[u]]:
                        pair_map2[v, pair_cnt2[v]][0] = w
                        pair_map2[v, pair_cnt2[v]][1] = u
                        pair_cnt2[v] += 1
                    for w in pair_map[v, :pair_cnt[v]]:
                        pair_map2[u, pair_cnt2[u]][0] = w
                        pair_map2[u, pair_cnt2[u]][1] = v
                        pair_cnt2[u] += 1

                # print(pair_cnt2[:20])
                # return

                pairs = pairs_catalog[group_start[max(gn[2] - (err - 1), 0)]:
                                      group_start[min(gn[2] + err + 2, 999)], 1:]

                candidates = []
                for pair in pairs:
                    u, v = pair.astype(np.int32) + 1
                    for w in pair_map2[u, :pair_cnt2[u]]:
                        if w[0] == v:
                            candidates.append([v, w[1], u, i, j, k])
                    for w in pair_map2[v, :pair_cnt2[v]]:
                        if w[0] == u:
                            candidates.append([u, w[1], v, i, j, k])
                # print(candidates)
                # print(len(candidates))                
                # return candidates
                keep_index = [1] * len(candidates)
                if len(candidates) > 0:
                    for i in range(len(candidates)):
                        candidate = candidates[i]
                        ww_cal = f_calibration(centers[candidate[3:], :], params)
                        t1i = ww_cal[1, :]
                        v2i = ww_cal[2, :]
                        t = np.array([t1i[1] * v2i[2] - t1i[2] * v2i[1], 
                                    t1i[2] * v2i[0] - t1i[0] * v2i[2], 
                                    t1i[0] * v2i[1] - t1i[1] * v2i[0]]).T
                        p1 = ww_cal[0, :].dot(t)
                        t1i = catalog[candidate[1], 1: 4]
                        v2i = catalog[candidate[2], 1: 4]
                        t = np.array([t1i[1] * v2i[2] - t1i[2] * v2i[1],
                                    t1i[2] * v2i[0] - t1i[0] * v2i[2],
                                    t1i[0] * v2i[1] - t1i[1] * v2i[0]])
                        p2 = catalog[candidate[0], 1: 4].dot(t)
                        
                        if p1 * p2 < 0:
                            keep_index[i] = 0
                            continue

                        d1 = np.arccos(catalog[candidate[0], 1: 4] * catalog[candidate[1], 1 : 4].T) - dis[candidate[3], candidate[4]]
                        d2 = np.arccos(catalog[candidate[1], 1: 4] * catalog[candidate[2], 1 : 4].T) - dis[candidate[4], candidate[5]]
                        d3 = np.arccos(catalog[candidate[2], 1: 4] * catalog[candidate[0], 1 : 4].T) - dis[candidate[5], candidate[3]]
                        candidate3s.append(np.hstack((candidate, [np.linalg.norm([d1, d2, d3])])))
                
                candidates = np.array(candidates)[np.where(keep_index)]
                
                for candidate in candidates:
                    candv = verify_pyramid(candidate, centers, params, err, pairs_catalog, catalog, group_cnt, group_start)
                    if candv.shape[0] == 0:
                        continue

                    for k in range(candv.shape[0]):
                        ww_cal = f_calibration(centers[candv[:4], :], params).T
                        vv_cal = catalog[candv[4: 8], 1: 4]
                        att, q = quest(ww_cal, vv_cal)
                        if not np.all(np.isreal(q)):
                            continue
                        ww_cal = f_calibration(centers_origin[:, :2], params)
                        smap = verification(catalog, q, ww_cal, 0.2)
                        if np.sum(smap > 0) >= 6:
                            identified = True
                            pb = np.sum(smap > 0)
                            break
                    if identified:
                        break
                if identified:
                    break
            if identified:
                break
        if identified:
            break
    if not identified:
        smap = - np.ones(centers_origin.shape[0])
        pb = 0
    
    return smap, pb, q, att
    
def reProjection(src, attitude, params):

    # 读星库
    sao60 = np.loadtxt('sao60.txt', dtype = float)

    # 各项参数
    h, w = 2048, 2048
    cx, cy, dx, dy, fov = params
    f = (cx * dx) / np.tan(fov / 2 * np.pi / 180)

    # 角度转换为弧度制
    ra, dec, rol = np.array(attitude) * np.pi / 180

    # 姿态转换矩阵：天球坐标系 -> 星敏感器坐标系
    r11 = - np.cos(rol) * np.sin(ra) - np.sin(rol) * np.sin(dec) * np.cos(ra)
    r12 = np.cos(rol) * np.cos(ra) - np.sin(rol) * np.sin(dec) * np.sin(ra)
    r13 = np.sin(rol) * np.cos(dec)
    r21 = np.sin(rol) * np.sin(ra) - np.cos(rol) * np.sin(dec) * np.cos(ra)
    r22 = - np.sin(rol) * np.cos(ra) - np.cos(rol) * np.sin(dec) * np.sin(ra)
    r23 = np.cos(rol) * np.cos(dec)
    r31 = np.cos(dec) * np.cos(ra)
    r32 = np.cos(dec) * np.sin(ra)
    r33 = np.sin(dec)

    Rbc = np.array([[r11, r12, r13], 
                    [r21, r22, r23], 
                    [r31, r32, r33]])
    
    # 姿态转换矩阵：星敏感器坐标系 -> 天球坐标系
    Rcb = Rbc.T

    # 视轴指向
    S = Rcb.dot(np.array([0, 0, 1]).T)

    # 所有星点的天球坐标系下的坐标
    allStar = sao60[:, 1: 4]

    # 所有星点方向与视轴方向的夹角
    allDist = np.arccos(allStar.dot(S))

    # 将天球坐标系转换到星敏感器坐标系
    allStar = Rbc.dot(allStar.T)

    # 过滤出投影在图像中的星点并保存其相关信息
    cnt = 0

    starInSky = np.zeros((500, 7))
    for i in range(sao60.shape[0]):
        if allDist[i] < 0.75 * fov * np.pi / 180:
            star = allStar[:, i]
            x = - f * star[0] / star[2] / dx + cx
            y = - f * star[1] / star[2] / dy + cy - 1024
            if x > 0 and x < src.shape[1] and y > 0 and y < src.shape[0]:
                starInSky[cnt, :5] = sao60[i, :5]
                starInSky[cnt, 5:] = [x, y]
                cnt += 1
    starInSky = starInSky[:cnt, :]
  
    # 建立图像
    resImg = Image.fromarray(src)
    font = ImageFont.truetype('C:\\Windows\\Fonts\\SIMYOU.TTF', 24)
    anno = ImageDraw.Draw(resImg)
    
    # 画星点和标注星等信息
    for star in starInSky:
        anno.ellipse((star[5] - 6, star[6] - 6, star[5] + 6, star[6] + 6), fill = 'white')
        anno.text((star[5] + 10, star[6] + 10), '{:.2f}'.format(star[4]), font = font, fill = 'white')
        
    resImg = np.array(resImg)

    # 返回图像
    return resImg        

if __name__ == "__main__":
    
    # 满足20度焦距的所有星对 (406387 x 3)
    dist20final = np.load('./params/dist20final.npy')
    
    # sao60星表归一化 (4931 x 9)
    sao60_uniform = np.load('./params/sao60_uniform.npy') 
    
    # 将 dist20final 分成 1000 份之后，各个片段的起始位置 （1000 x 1）
    group_start = np.load('./params/group_start.npy').astype(np.int32)
    
    # 将 dist20final 分成 1000 份之后，各个片段的个数 (1000 x 1)
    group_cnt = np.load('./params/group_cnt.npy').astype(np.int32)

    # 星敏感器参数 （主点、像元尺寸、焦距等）
    params = np.array([1024, 1024, 0.0055, 0.0055, 25, 0, 0, 0, 0])
    
    # 读取质心
    centers = np.load('./params/centers.npy')
    centers[:, 1] += 1024

    res = pyramid(centers, params, 1, 6, sao60_uniform, dist20final, group_start, group_cnt)
    print(res[-1] * 180 / np.pi)
    img = np.zeros((1024, 2048))
    resImg = reProjection(img, res[-1] * 180 / np.pi, [1024, 1024, 0.0055, 0.0055, 25])
    plt.figure()
    plt.imshow(resImg, cmap='gray', vmin=0, vmax=255)
    plt.show()