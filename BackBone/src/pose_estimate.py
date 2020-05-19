def detect_keypoint(n, outputs):
    maxs = -3
    maxc = []
    for i in range(9):
        for j in range(9):
            if(outputs[0][0][i][j][n] > maxs):
                maxs = outputs[0][0][i][j][n]
                maxc = [i, j]
    if maxs > -3:
        y = int(maxc[0] * 32 + outputs[1][0][maxc[0]][maxc[1]][n])
        x = int(maxc[1] * 32 + outputs[1][0][maxc[0]][maxc[1]][n + 17])
    else:
        return -1,-1
    return x, y