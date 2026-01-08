-module(perlang).
-export([main/0]).

main() ->
    {ok, File} = file:open("../data/Erlang.txt", [write]),
    generate(1000000, File),
    file:close(File),
    init:stop().

generate(0, _) -> ok;
generate(1, File) ->
    Num = rand:uniform(),
    io:format(File, "~.10f", [Num]);
generate(N, File) ->
    Num = rand:uniform(),
    io:format(File, "~.10f~n", [Num]),
    generate(N - 1, File).