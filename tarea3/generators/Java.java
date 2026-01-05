import java.util.Random;
import java.io.FileWriter;
import java.io.IOException;

public final class Java {
    private static final String FILE_PATH = "../data/Java.txt";
    private static final int N = 1000000;

    private Java() {}

    public static void main(String[] args) {
        try (FileWriter fw = new FileWriter(FILE_PATH)) {
            final Random random = new Random();
           
            for (int i = 0; i < N; i++) {
                fw.write(Double.toString(random.nextDouble()));
                fw.write('\n');
            }

            fw.close(); 
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}