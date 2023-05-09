import mongoose from 'mongoose';
import request from 'supertest';
import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import app from '../index';

import { INews, News } from '../model/News';