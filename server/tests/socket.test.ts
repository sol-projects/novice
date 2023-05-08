import { Server } from 'http';
import { AddressInfo } from 'net';
import { io as Client } from 'socket.io-client';
import app from '../index';
import { init } from '../socket/socket';

describe('Socket.io tests', () => {
  let server: Server;
  let socket: Socket;

  beforeAll(() => {
    server = app.listen();
    init(server);
  });

  afterAll(() => {
    server.close();
  });

  beforeEach((done) => {
    const address = server.address() as AddressInfo;
    const url = `http://localhost:${address.port}`;
    socket = Client(url);
    socket.on('connect', () => {
      done();
    });
  });

  afterEach(() => {
    socket.disconnect();
  });

  test('connection test', (done) => {
    expect(socket.connected).toBeTruthy();
    done();
  });

  test('message test', (done) => {
    const testMessage = 'Hello, World!';

    socket.on('message', (message: string) => {
      expect(message).toBe(testMessage);
      done();
    });

    socket.emit('message', testMessage);
  });
});
