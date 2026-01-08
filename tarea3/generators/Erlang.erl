-module(erlang).
-export([main/0]).

main() ->
    {ok, File} = file:open("../data/Erlang.txt", [write]),
    generate(1000000, File),
    file:close(File),
    init:stop().

generate(0, _) -> ok;
generate(N, File) ->
    % rand:uniform() genera un float entre 0.0 y 1.0
    Num = rand:uniform(),
    io:format(File, "~.10f~n", [Num]),
    generate(N - 1, File).