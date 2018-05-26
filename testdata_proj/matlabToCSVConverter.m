for i = (1:9)
  [s, h] = sload(strcat('A0', int2str(i), 'E.gdf'));
  csvwrite(strcat('A0', int2str(i), 'EInstances.csv'), s)
  csvwrite(strcat('A0', int2str(i), 'ELabels.csv'), [h.EVENT.TYP, h.EVENT.POS, h.EVENT.DUR])
end