import sys
import csv
sys.path.append('/content/Mask-CycleGAN/FID/src')
from pytorch_fid.fid_score import main



def grid_with_id(grid, ids):
  N = len(ids)
  assert len(grid) == N
  out_grid = [[''] * (N + 1) for _ in range(N + 1)]

  # Write ids at top row.
  out_grid[0][1:] = list(ids)

  for i in range(1, N + 1):
    # Write ids at left column.
    out_grid[i][0] = ids[i - 1]
    out_grid[i][1:] = list(grid[i - 1])

  return out_grid



def compute_fid_matrix(name, folder, scale50, scale80, scale100, train, test):

    data_ids = [
          'scale=0.50',
          'scale=0.80',
          'scale=1.00',
          'train',
          'test',
      ]
    
    fid_grid = [[0] * 5 for _ in range(5)]
    
    for i in range(5):
    
      path1 = ""
      if(i==0):
        path1 = scale50
      if(i==1):
        path1 = scale80
      if(i==2):
        path1 = scale100
      if(i==3):
        path1 = train
      if(i==4):
        path1 = test
      for j in range(i + 1, 5):
        path2 = ""
        if(j==0):
          path2 = scale50
        if(j==1):
          path2 = scale80
        if(j==2):
          path2 = scale100
        if(j==3):
          path2 = train
        if(j==4):
          path2 = test
    
        fid_value = main(path1, path2)
        fid_grid[i][j] = fid_value
        fid_grid[j][i] = fid_value
        print(path1+"     "+path2)
        print(fid_value)
    
    fid_grid = grid_with_id(fid_grid, data_ids)
    print(fid_grid)
    for row in fid_grid:
        print(row)
    
    with open(folder+"/FID_Matrix-"+name, 'w') as fout:
          writer = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
          writer.writerows(fid_grid)
