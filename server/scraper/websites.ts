import { INews } from '../model/News';

export const websites = new Map<string, (n: number) => Promise<INews[]>>([
  ['siol', require('./websites/siol')],
  ['gov novice', require('./websites/gov')],
  //['gov novice vlade', require('./websites/gov_vlada')],
  ['24 ur', require('./websites/24ur')],
  //['delo', require('./websites/delo')],
]);
