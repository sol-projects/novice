import { INews } from '../model/News';

export const websites = new Map<string, (n: number) => Promise<INews[]>>([
  ['gov novice', require('./websites/gov')],
  ['gov novice vlade', require('./websites/gov_vlada')],
  ['24 ur', require('./websites/24ur')],
  //['siol', require('./websites/siol')],
  //['delo', require('./websites/delo')],
]);
