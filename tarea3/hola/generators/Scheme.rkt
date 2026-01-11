#lang racket

(define N 1000000)
(define file-path "../data/Scheme.txt")

(define (generate-numbers)
  (make-parent-directory* file-path)
  (call-with-output-file file-path
    (lambda (out)
      (for ([i (in-range N)])
        (when (> i 0)
          (newline out))
        (display (+ 1 (random 20)) out)))
    #:exists 'replace))

(generate-numbers)