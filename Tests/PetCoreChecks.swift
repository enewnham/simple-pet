import Darwin
import Foundation

@main
@MainActor
struct PetCoreChecks {
    static func main() async throws {
        let checks = PetCoreChecks()
        try checks.testRejectsMalformedEventsWithoutApplyingPartialState()
        try await checks.testSpeechExpiresWithoutChangingState()
        try await checks.testNewSpeechSurvivesEarlierExpirationAndStateInputPreservesIt()
        try await checks.testSpeechAcceptanceRequiresVisibleTextAndRetriesPreserveCurrentSpeech()
        try checks.testNapCommandsPreserveActivityAndWakeIndependently()
        try await checks.testSocketReceivesValidEventsAfterInvalidJSON()
        try await checks.testSocketReceivesLongSpeechWithoutTruncation()
        try checks.testReclaimsStaleSocketAndPreservesRegularFiles()
        try checks.testFinitePlaybackHoldsLastFrameAndCompletesOnce()
        try checks.testMotionStaysWithinMonitorBounds()
        try checks.testRejectsInvalidSpritePacks()
        print("Pet core checks passed (11 scenarios).")
    }
    private func event(_ json: String) throws -> PetEvent {
        try PetEvent.decode(Data(json.utf8))
    }

    func testRejectsMalformedEventsWithoutApplyingPartialState() throws {
        for json in ["{}", "{\"state\":\"busy\"}", "{\"state\":\"idle\",\"ttl\":-1}",
                     "{\"text\":false}", "{\"animation\":{\"id\":\"jump\",\"command_id\":1,\"repeats\":0}}",
                     "{\"motion\":{\"x\":0,\"y\":0,\"seconds\":0,\"command_id\":2}}"] {
            try requireThrows { try event(json) }
        }
        try require(try event("{\"text\":\"\(String(repeating: "x", count: 4096))\"}").text?.count == 4096)
        try require(try event("{\"state\":\"working\"}").state == .working)
    }

    func testSpeechExpiresWithoutChangingState() async throws {
        let model = PetModel()
        model.apply(try event("{\"state\":\"working\",\"text\":\"Working\",\"ttl\":0.03}"))
        try require(model.text == "Working")
        try await waitUntil { model.text == nil }
        try require(model.state == .working)
    }

    func testNewSpeechSurvivesEarlierExpirationAndStateInputPreservesIt() async throws {
        let model = PetModel()
        model.apply(try event("{\"state\":\"working\",\"text\":\"First\",\"ttl\":0.02}"))
        model.apply(try event("{\"text\":\"Second\",\"ttl\":1}"))
        try await Task.sleep(nanoseconds: 60_000_000)
        try require(model.text == "Second")
        try require(model.state == .working)
        model.apply(try event("{\"state\":\"idle\"}"))
        try require(model.text == "Second")
        try require(model.state == .idle)
        model.apply(try event("{\"text\":\"\"}"))
        try require(model.text == nil)
    }

    func testSpeechAcceptanceRequiresVisibleTextAndRetriesPreserveCurrentSpeech() async throws {
        let model = PetModel()
        let identifier = UUID().uuidString
        let speech = try event("{\"text\":\"Accepted\",\"ttl\":0.02,\"delivery_id\":\"\(identifier)\"}")
        try require(model.apply(speech) == identifier)
        try require(model.text == "Accepted")
        try await waitUntil { model.text == nil }
        try require(model.apply(speech) == identifier)
        try require(model.text == nil)
        model.apply(try event("{\"text\":\"New speech\",\"ttl\":1}"))
        try require(model.apply(speech) == identifier)
        try require(model.text == "New speech")
        for input in ["{\"text\":\"   \",\"ttl\":1}", "{\"text\":\"Hidden\",\"ttl\":0}", "{\"state\":\"working\"}"] {
            let data = input.dropLast() + ",\"delivery_id\":\"\(UUID().uuidString)\"}"
            try require(model.apply(try event(String(data))) == nil)
        }
        try requireThrows { try event("{\"text\":\"Rejected\",\"delivery_id\":\"invalid\"}") }
    }

    func testSocketReceivesValidEventsAfterInvalidJSON() async throws {
        let path = "/tmp/pet-test-\(UUID().uuidString).sock"
        var received: [PetEvent] = []
        let server = try SocketServer(path: path) { event in
            received.append(event)
        }
        defer { server.stop() }
        let client = try makeClient(path: path)
        defer { close(client) }
        try send(Data("invalid JSON".utf8), to: client)
        try send(Data("{\"state\":\"error\",\"text\":\"Quoted \\\"text\\\" 🐱\"}".utf8), to: client)
        try await waitUntil { !received.isEmpty }
        try require(received.count == 1)
        try require(received[0].state == .error)
        try require(received[0].text == "Quoted \"text\" 🐱")
        var info = stat()
        try require(lstat(path, &info) == 0)
        try require(info.st_mode & 0o777 == 0o600)
        try requireThrows { try SocketServer(path: path) { _ in } }
    }

    func testNapCommandsPreserveActivityAndWakeIndependently() throws {
        let model = PetModel()
        model.apply(try event("{\"sleeping\":true}"))
        try require(model.sleeping && model.state == .idle)
        model.apply(try event("{\"state\":\"idle\"}"))
        try require(model.sleeping)
        model.apply(try event("{\"sleeping\":false,\"state\":\"working\"}"))
        try require(!model.sleeping && model.state == .working)
        try requireThrows { try event("{\"sleeping\":\"true\"}") }
    }

    func testSocketReceivesLongSpeechWithoutTruncation() async throws {
        let path = "/tmp/pet-test-\(UUID().uuidString).sock"
        var received: [PetEvent] = []
        let server = try SocketServer(path: path) { received.append($0) }
        defer { server.stop() }
        let client = try makeClient(path: path)
        defer { close(client) }
        let speech = String(repeating: "Unicode speech 🐱. ", count: 1000)
        let payload = try JSONSerialization.data(withJSONObject: ["text": speech])
        try require(payload.count > 16384)
        try send(payload, to: client)
        try await waitUntil { !received.isEmpty }
        try require(received.count == 1)
        try require(received[0].text == speech)
    }

    func testFinitePlaybackHoldsLastFrameAndCompletesOnce() throws {
        let command = try event("{\"animation\":{\"id\":\"jump\",\"repeats\":3,\"command_id\":7}}").animation!
        var playback = AnimationPlayback(commandID: command.commandID, clipID: command.id, count: 5, fps: 10,
                                         repeats: command.repeats, start: 100)
        try require(playback.commandID == 7)
        try require(playback.sample(at: 100.6).frame == 0)
        try require(playback.sample(at: 101.4).frame == 4)
        let finished = playback.sample(at: 101.5)
        try require(finished.frame == 4 && finished.completed)
        try require(!playback.sample(at: 105).completed)
        var loop = AnimationPlayback(commandID: 8, clipID: "rest", count: 2, fps: 2, repeats: nil, start: 0)
        try require(!loop.sample(at: 100).completed)
    }

    func testMotionStaysWithinMonitorBounds() throws {
        let command = try event("{\"motion\":{\"x\":2000,\"y\":-500,\"seconds\":2,\"command_id\":9}}").motion!
        let size = CGSize(width: 200, height: 160)
        let bounds = CGRect(x: -100, y: 20, width: 1000, height: 700)
        let motion = WindowMotion(command: command, origin: CGPoint(x: 100, y: 100), size: size, bounds: bounds, started: 10)
        let middle = motion.sample(at: 11, size: size, bounds: bounds)
        try require(middle.origin == CGPoint(x: 400, y: 60) && !middle.finished)
        let final = motion.sample(at: 12, size: size, bounds: bounds)
        try require(final.origin == CGPoint(x: 700, y: 20) && final.finished)
        let smaller = motion.sample(at: 12, size: size, bounds: CGRect(x: 0, y: 0, width: 100, height: 100))
        try require(smaller.origin == .zero)
    }

    func testRejectsInvalidSpritePacks() throws {
        let directory = URL(fileURLWithPath: "/tmp/pet-pack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data([0]).write(to: directory.appendingPathComponent("rest.png"))
        let manifest = directory.appendingPathComponent("manifest.json")
        let valid = "{\"pet\":\"quiet\",\"rest\":\"rest\",\"clips\":{\"rest\":{\"frames\":[\"rest.png\"],\"fps\":1,\"role\":\"rest\"}}}"
        try Data(valid.utf8).write(to: manifest)
        try require(try SpriteManifest.load(directory: directory).clips.count == 1)
        for invalid in [valid.replacingOccurrences(of: "rest.png", with: "missing.png"),
                        valid.replacingOccurrences(of: "rest.png", with: "../rest.png"),
                        valid.replacingOccurrences(of: "\"fps\":1", with: "\"fps\":0"),
                        valid.replacingOccurrences(of: "\"rest\":\"rest\"", with: "\"rest\":\"missing\"")] {
            try Data(invalid.utf8).write(to: manifest)
            try requireThrows { try SpriteManifest.load(directory: directory) }
        }
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(2)
        while !condition() {
            guard Date() < deadline else { throw NSError(domain: "TestTimeout", code: 1) }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    func testReclaimsStaleSocketAndPreservesRegularFiles() throws {
        let path = "/tmp/pet-test-\(UUID().uuidString).sock"
        var address = makeAddress(path)
        let stale = socket(AF_UNIX, SOCK_DGRAM, 0)
        try require(stale >= 0)
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(stale, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        close(stale)
        try require(result == 0)
        let server = try SocketServer(path: path) { _ in }
        server.stop()
        server.stop()
        var removed = stat()
        try require(lstat(path, &removed) == -1 && errno == ENOENT)
        let file = "/tmp/pet-test-\(UUID().uuidString).txt"
        try Data("Keep me".utf8).write(to: URL(fileURLWithPath: file))
        defer { try? FileManager.default.removeItem(atPath: file) }
        try requireThrows { try SocketServer(path: file) { _ in } }
        try require(try String(contentsOfFile: file, encoding: .utf8) == "Keep me")
    }

    private func makeAddress(_ path: String) -> sockaddr_un {
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: Array(path.utf8) + [0]) }
        return address
    }

    private func makeClient(path: String) throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_DGRAM, 0)
        var sendBuffer: Int32 = 131072
        guard setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sendBuffer, socklen_t(MemoryLayout<Int32>.size)) == 0 else {
            close(fd)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        var address = makeAddress(path)
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard result == 0 else { close(fd); throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        return fd
    }

    private func send(_ data: Data, to fd: Int32) throws {
        let sent = data.withUnsafeBytes { Darwin.send(fd, $0.baseAddress, $0.count, 0) }
        guard sent == data.count else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
    }
}

private struct CheckFailure: Error { let message: String }

private func require(_ condition: @autoclosure () throws -> Bool) throws {
    if try !condition() { throw CheckFailure(message: "Assertion failed") }
}

private func requireThrows<T>(_ operation: () throws -> T) throws {
    do {
        _ = try operation()
    } catch {
        return
    }
    throw CheckFailure(message: "Expected operation to fail")
}
