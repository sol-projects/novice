import socketIO, { Server, Socket } from 'socket.io';

let io: Server;

export function init(server: any) {
  io = new Server(server);

  io.on('connection', (socket) => {
    console.log(`socket ${socket.id} connected`);

    socket.on('message', (message) => {
      console.log(`message from ${socket.id} : ${message}`);
    });

    socket.on('disconnect', () => {
      console.log(`socket ${socket.id} disconnected`);
    });
  });
}

export function emit(eventName: string, data: any): void {
  io.emit(eventName, data);
}
