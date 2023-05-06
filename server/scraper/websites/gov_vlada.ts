import cheerio from 'cheerio';
import axios from 'axios';
import { INews } from '../../model/News';
import gov_shared from './gov_shared';

async function _gov_vlada(n: number) {
  return gov_shared(n, 'https://www.gov.si/drzavni-organi/vlada/novice/');
}

export = _gov_vlada;
