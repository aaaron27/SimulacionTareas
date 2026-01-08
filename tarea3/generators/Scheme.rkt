#lang racket

(define N 1000000)
(define file-path "../data/Scheme.txt")

(define (generate-numbers)
  (call-with-output-file file-path
    (lambda (out)
      (for ([i N])
        (displayln (+ 1 (random 20)) out)))
    #:exists 'replace))

(generate-numbers)