# Local-feature twin verification (discoveries #22 follow-up). Some no-twin test rows are copies rotated by odd angles and heavily noised,
# which global descriptors (SSCD/DINOv2) miss; SIFT (rotation-invariant) + ratio test + RANSAC homography inliers can verify them.
# Candidates per query = top-10 SSCD + top-10 DINOv2 train neighbours. Queries = no-twin clean_other rows (test, plus train for calibration).
# -> work/sift_pairs.npz (query, cand, inliers). CPU, multiprocessing.
import sys, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
sys.path.insert(0, 'work'); from twin import M, TR, ST, vote
MAXS, NKP = 448, 600

def feats(i):  # denoise hard (copies can carry heavy noise), grayscale, max side 448, top-600 SIFT keypoints -> uint8 descriptors
    a = cv2.imread(M.path[i]); a = cv2.GaussianBlur(cv2.medianBlur(a, 5), (0, 0), 1.0); g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    s = MAXS / max(g.shape); g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    kp, d = cv2.SIFT_create(nfeatures=NKP).detectAndCompute(g, None)
    return i, (np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)), (np.clip(d, 0, 255).astype(np.uint8) if d is not None else np.zeros((0, 128), np.uint8))

def match(fq, fc):
    (pq, dq), (pc, dc) = fq, fc
    if len(dq) < 8 or len(dc) < 8: return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dq.astype(np.float32), dc.astype(np.float32), k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 8: return len(good) // 2
    _, mask = cv2.findHomography(pq[[g.queryIdx for g in good]], pc[[g.trainIdx for g in good]], cv2.RANSAC, 6.0)
    return int(mask.sum()) if mask is not None else 0

def job(task):  # one query + its candidates per task: features computed in the worker, nothing large is shared (memory-safe)
    q, cands = task; fq = feats(q)[1:]
    return [(q, c, match(fq, feats(c)[1:])) for c in cands]

if __name__ == '__main__':
    S1, S2 = np.load('work/nntr_sscd.npz'), np.load('work/nntr_dinob.npz'); lab, _ = vote(S1['i'], S1['s'], 0.55)
    Q = np.where((lab < 0) & (ST == 'clean_other'))[0]  # train + test rows without an SSCD twin
    tasks = [(int(q), sorted({int(c) for c in np.r_[S1['i'][q], S2['i'][q]]})) for q in Q]
    print('queries', len(Q), '(test', int((~TR[Q]).sum()), ') pairs', sum(len(c) for _, c in tasks), flush=True)
    out = []
    with Pool(6) as pool:
        for k, r in enumerate(pool.imap_unordered(job, tasks, chunksize=4)):
            out += r
            if k % 300 == 0: print(k, len(tasks), flush=True)
    P = np.array(out); np.savez('work/sift_pairs.npz', q=P[:, 0], c=P[:, 1], inl=P[:, 2]); print('saved', len(P), flush=True)
