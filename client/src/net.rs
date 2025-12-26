use std::{
    error::Error,
    net::{SocketAddr, UdpSocket},
    sync::mpsc::{Receiver, Sender},
};

use shared::net::Packet;

pub struct NetworkClient {
    receive: std::sync::mpsc::Receiver<Packet>,
    send: std::sync::mpsc::Sender<Packet>,
}

impl NetworkClient {
    pub fn connect(
        recieve_addr: SocketAddr,
        send_addr: SocketAddr,
    ) -> Result<Self, Box<dyn Error>> {
        let send_socket = UdpSocket::bind(recieve_addr)?;
        send_socket.connect(send_addr)?;

        let (send_tx, send_rx): (Sender<Packet>, Receiver<Packet>) = std::sync::mpsc::channel();
        let (receive_tx, receive_rx) = std::sync::mpsc::channel();

        let receive_socket = send_socket.try_clone()?;
        let mut receive_buffer = vec![0; 2000];
        std::thread::spawn(move || {
            loop {
                let Ok(size) = receive_socket.recv(&mut receive_buffer) else {
                    continue;
                };

                let packet = Packet::deserialize(&receive_buffer[0..size]);

                if let Err(e) = receive_tx.send(packet) {
                    continue;
                }
            }
        });

        let mut send_buffer = vec![0; 2000];
        std::thread::spawn(move || {
            loop {
                let Ok(packet) = send_rx.recv() else {
                    continue;
                };

                packet.serialize(&mut send_buffer);

                if let Err(e) = send_socket.send(&send_buffer) {
                    continue;
                }
            }
        });

        Ok(Self {
            receive: receive_rx,
            send: send_tx,
        })
    }

    pub fn send(&mut self, packet: Packet) -> Result<(), Box<dyn Error>> {
        self.send.send(packet)?;

        Ok(())
    }

    pub fn receive(&mut self) -> Result<Packet, Box<dyn Error>> {
        let packet = self.receive.try_recv()?;

        Ok(packet)
    }
}
