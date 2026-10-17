import Foundation
import NaturalLanguage

struct Request: Decodable {
    let text: String
    let previous: [String]
}

struct Response: Encodable {
    let available: Bool
    let similarities: [Double?]
}

func similarity(_ first: [Double], _ second: [Double]) -> Double? {
    guard first.count == second.count else { return nil }
    var dot = 0.0
    var firstNorm = 0.0
    var secondNorm = 0.0
    for (left, right) in zip(first, second) {
        dot += left * right
        firstNorm += left * left
        secondNorm += right * right
    }
    guard firstNorm > 0, secondNorm > 0 else { return nil }
    let value = dot / sqrt(firstNorm * secondNorm)
    return value.isFinite ? min(1, max(-1, value)) : nil
}

do {
    let request = try JSONDecoder().decode(Request.self, from: FileHandle.standardInput.readDataToEndOfFile())
    let embedding = NLEmbedding.sentenceEmbedding(for: .english)
    let vector = embedding?.vector(for: request.text)
    let values = request.previous.map { text -> Double? in
        guard let vector, let previous = embedding?.vector(for: text) else { return nil }
        return similarity(vector, previous)
    }
    let response = Response(available: embedding != nil && vector != nil, similarities: values)
    FileHandle.standardOutput.write(try JSONEncoder().encode(response))
} catch {
    FileHandle.standardError.write(Data("Invalid similarity request\n".utf8))
    exit(1)
}
