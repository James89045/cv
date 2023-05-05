from HW2 import read_img
from HW2 import img_to_gray
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm
path = "bonus/"
m1, m1_gray = read_img(path + "m2.jpg")
m2, m2_gray = read_img(path + "m3.jpg")
m = [m1, m2]
m1_kp, m1_des = cv2.SIFT_create().detectAndCompute(m1_gray, None)
m2_kp, m2_des = cv2.SIFT_create().detectAndCompute(m2_gray, None)
#print(m1_kp[0].pt[0], m1_kp[0].pt[1])
def match_kp(l_des, r_des):
    matchidanddist = []
    for i in range(len(l_des)):
        min_idanddist = [-1, np.inf]
        secmin_idanddist = [-1, np.inf]
        for j in range(len(r_des)):
            dist = np.linalg.norm(l_des[i] - r_des[j])
            if (dist < min_idanddist[1]):
                secmin_idanddist[1] = min_idanddist[1]
                min_idanddist = [j, dist]
            elif (dist < secmin_idanddist[1] and secmin_idanddist[1] > min_idanddist[1]):
                secmin_idanddist = [j, dist]
        matchidanddist.append([min_idanddist[0], min_idanddist[1], secmin_idanddist[0], secmin_idanddist[1]])

    return matchidanddist#list中每個index所含的資料代表的是在m1_des對應的m2_des前兩近的vector的index以及距離大小

def Lowetest(matchidanddist, m1_kp, m2_kp, ratio):
    # 確認list內有哪一些點可以使用
    goodmatch = []
    useful_kp = []
    for i in range(len(matchidanddist)):
        if (matchidanddist[i][1] <= matchidanddist[i][3]*ratio):
            #儲存goodmatch的index以及相對應的index
            goodmatch.append([i, matchidanddist[i][0]])

    for m1_id, m2_id in goodmatch:
        m1_loc = [int(m1_kp[m1_id].pt[0]), int(m1_kp[m1_id].pt[1])]
        m2_loc = [int(m2_kp[m2_id].pt[0]), int(m2_kp[m2_id].pt[1])]
        useful_kp.append([m1_loc, m2_loc])
    
    useful_kp = np.array(useful_kp)
    # print(len(useful_kp))
    return useful_kp


def drawMatches(imgs, matches_pos):
        
    # initialize the output visualization image
    img_left, img_right = imgs
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]
    vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
    vis[0:hl, 0:wl] = img_left
    vis[0:hr, wl:] = img_right
    
    # Draw the match
    for (img_left_pos, img_right_pos) in matches_pos:
            pos_l = img_left_pos
            pos_r = img_right_pos[0] + wl, img_right_pos[1]
            cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
            cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
            cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)
            
    # return the visualization
    plt.figure(1)
    plt.title("img with matching points")
    plt.imshow(vis[:,:,::-1])
    plt.show()
    #cv2.imwrite("Feature matching img/matching.jpg", vis)
    
    return vis


def homography(pair):
    A = []
    for i in range(len(pair)):
        A.append([-pair[i, 0, 0], -pair[i, 0, 1], -1, 0, 0, 0, pair[i, 1, 0]*pair[i, 0, 0], pair[i, 1, 0]*pair[i, 0, 1], pair[i, 1, 0]])
        A.append([0, 0, 0, -pair[i, 0, 0], -pair[i, 0, 1], -1, pair[i, 1, 1]*pair[i, 0, 0], pair[i, 1, 1]*pair[i, 0, 1], pair[i, 1, 1]])

    A = np.array(A)
    U, s, V = np.linalg.svd(A)
    H = np.reshape(V[8], (3,3))
    H = H/H[2][2]
    return H


class best_homo():
    def __init__(self):
        pass

    def random_4prs(self, all_pairs):
        id = random.sample(range(len(all_pairs)), 4)
        select = [all_pairs[i] for i in id]
        select = np.array(select)
        return select

    def error(self, input, H):
        left = input[:, 0]
        right = input[:, 1].T
        # print(left)
        left = np.concatenate((left, np.ones((len(input), 1))), axis=1).T
        est_right = np.dot(H, left)
        # print(est_right)
        for i in range(est_right.shape[1]):
            est_right[:, i] /= est_right[2, i]
        est_right = est_right[0:2]
        # print(est_right.shape)
        # print(right.shape)
        all_error = np.linalg.norm(right - est_right, axis=0) ** 2
        # print(all_error)
        return all_error
    
    def ransac(self, matches, threshold, iters):
        num_best_inliers = 0
        
        for i in range(iters):
            points = self.random_4prs(matches)
            H = homography(points)
            
            #  avoid dividing by zero 
            if np.linalg.matrix_rank(H) < 3:
                continue
                
            errors = self.error(matches, H)
            idx = np.where(errors < threshold)[0]
            inliers = matches[idx]

            num_inliers = len(inliers)
            if num_inliers > num_best_inliers:
                best_inliers = inliers.copy()
                num_best_inliers = num_inliers
                best_H = H.copy()
                
        print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
        return best_inliers, best_H

        
def stitch_img(left, right, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)
    print("left: ", left)   
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image


matching = match_kp(m1_des, m2_des)
t = Lowetest(matching, m1_kp, m2_kp, 0.7)
x, h = best_homo().ransac(t, 0.5, 3000)
img = stitch_img(m1, m2, h)
plt.imshow(img)
plt.show()