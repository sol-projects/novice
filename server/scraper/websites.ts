import { INews } from '../model/News';

const websites = new Map<string, (n: number) => Promise<INews[]>>([
  ['siol', require('./websites/siol')],
  ['gov', require('./websites/gov')],
  ['gov-vlada', require('./websites/gov_vlada')],
  ['24ur', require('./websites/24ur')],
  ['delo', require('./websites/delo')],
]);

export = websites;
