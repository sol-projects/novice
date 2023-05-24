import { INews } from '../model/News';

const websites = new Map<string, (n: number) => Promise<INews[]>>([
  ['siol', require('./websites/siol')],
  ['gov', require('./websites/gov')],
  ['gov-vlada', require('./websites/gov_vlada')],
  ['24ur', require('./websites/24ur')],
  ['delo', require('./websites/delo')],
  ['mbinfo', require('./websites/mariborinfo')],
  ['rtvslo', require('./websites/rtvSlo')],
  ['sta', require('./websites/servisSta')],
  //['ekipa24', require('./websites/ekipa24')], NI V REDU - lahko crasha mongodb
  //['dnevnik', require('./websites/dnevnik')], NI V REDU - lahko crasha server
  //['svet24', require('./websites/svet24')], NI V REDU - lahko crasha server
  //['n1info', require('./websites/n1info')], NI V REDU - nepravilen info
]);

export = websites;
