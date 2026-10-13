import Darwin
import Foundation

public final class SocketServer {
    public static let maximumMessageBytes = 65536
    private var source: DispatchSourceRead?
    private let queue = DispatchQueue(label: "pet.socket")
    private let shutdown = DispatchGroup()

    public init(path: String, receive: @escaping (PetEvent) -> Void) throws {
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        let bytes = Array(path.utf8) + [0]
        guard !path.isEmpty, !path.utf8.contains(0), bytes.count <= MemoryLayout.size(ofValue: address.sun_path) else {
            throw SocketError(message: "Socket path is empty or exceeds the UNIX socket limit")
        }
        withUnsafeMutableBytes(of: &address.sun_path) { destination in
            destination.copyBytes(from: bytes)
        }
        var existing = stat()
        if lstat(path, &existing) == 0 {
            guard existing.st_mode & S_IFMT == S_IFSOCK, existing.st_uid == getuid() else {
                throw SocketError(message: "Refusing to remove a socket path owned by another user or a regular file")
            }
            let probe = socket(AF_UNIX, SOCK_DGRAM, 0)
            guard probe >= 0 else { throw SocketError.system("socket") }
            let connected = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.connect(probe, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
            let probeError = errno
            close(probe)
            guard connected != 0 else {
                throw SocketError(message: "An overlay is already listening at \(path)")
            }
            guard probeError == ECONNREFUSED || probeError == ENOENT else {
                throw SocketError(message: "Cannot determine whether the existing socket is stale")
            }
            guard unlink(path) == 0 || errno == ENOENT else { throw SocketError.system("unlink") }
        } else if errno != ENOENT {
            throw SocketError.system("lstat")
        }
        let fd = socket(AF_UNIX, SOCK_DGRAM, 0)
        guard fd >= 0 else { throw SocketError.system("socket") }
        var bound = false
        do {
            var receiveBuffer: Int32 = 131072
            guard setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receiveBuffer, socklen_t(MemoryLayout<Int32>.size)) == 0 else {
                throw SocketError.system("setsockopt")
            }
            guard fcntl(fd, F_SETFL, O_NONBLOCK) == 0,
                  fcntl(fd, F_SETFD, FD_CLOEXEC) == 0 else { throw SocketError.system("fcntl") }
            let result = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
            guard result == 0 else { throw SocketError.system("bind") }
            bound = true
            guard chmod(path, 0o600) == 0 else { throw SocketError.system("chmod") }
            var identity = stat()
            guard lstat(path, &identity) == 0 else { throw SocketError.system("lstat") }
            let device = identity.st_dev
            let inode = identity.st_ino
            let shutdown = self.shutdown
            let readSource = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
            readSource.setEventHandler {
                var buffer = [UInt8](repeating: 0, count: Self.maximumMessageBytes + 1)
                // Limit each drain so sustained producers cannot monopolize the reader.
                for _ in 0..<64 {
                    let count = recv(fd, &buffer, buffer.count, 0)
                    if count < 0 {
                        if errno == EINTR { continue }
                        break
                    }
                    guard count > 0, count <= Self.maximumMessageBytes else { continue }
                    let data = Data(buffer.prefix(count))
                    guard let event = try? PetEvent.decode(data) else { continue }
                    DispatchQueue.main.async { receive(event) }
                }
            }
            readSource.setCancelHandler {
                close(fd)
                var current = stat()
                if lstat(path, &current) == 0, current.st_dev == device, current.st_ino == inode {
                    unlink(path)
                }
                shutdown.leave()
            }
            shutdown.enter()
            source = readSource
            readSource.resume()
        } catch {
            close(fd)
            if bound { unlink(path) }
            throw error
        }
    }

    public func stop() {
        source?.cancel()
        shutdown.wait()
    }
    deinit { stop() }
}

private struct SocketError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
    static func system(_ operation: String) -> SocketError {
        SocketError(message: "\(operation): \(String(cString: strerror(errno)))")
    }
}

public final class SocketSender {
    private let fd: Int32
    private var address: sockaddr_un

    public init(path: String) throws {
        address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        let bytes = Array(path.utf8) + [0]
        guard !path.isEmpty, !path.utf8.contains(0), bytes.count <= MemoryLayout.size(ofValue: address.sun_path) else {
            throw SocketError(message: "Invalid daemon socket path")
        }
        withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: bytes) }
        fd = socket(AF_UNIX, SOCK_DGRAM, 0)
        guard fd >= 0 else { throw SocketError.system("socket") }
        guard fcntl(fd, F_SETFL, O_NONBLOCK) == 0, fcntl(fd, F_SETFD, FD_CLOEXEC) == 0 else {
            close(fd)
            throw SocketError.system("fcntl")
        }
    }

    public func send(_ payload: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return }
        withUnsafePointer(to: &address) { addressPointer in
            addressPointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketPointer in
                data.withUnsafeBytes { bytes in
                    _ = sendto(fd, bytes.baseAddress, bytes.count, 0, socketPointer, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
        }
    }
    deinit { close(fd) }
}
