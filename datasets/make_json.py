import argparse

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    input_json_list = []
    with open(args.input, mode='r', encoding='utf-8') as f_in:
        logger.info("Input file: %s", args.input)
        for sentence in f_in:
            input_json_list.append("{\"source\" : \"" + sentence.strip() + "\"}")
    
    logger.info("Loaded %d sentences in total", len(input_json_list))
    
    with open(args.output, mode='w', encoding='utf-8') as f_out:
        logger.info("Output file: %s", args.output)
        for line in input_json_list:
            f_out.write(line + '\n')
    
    logger.info("Writing Done.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-I', type=str)
    parser.add_argument('--output', '-O', type=str)
    args = parser.parse_args()
    
    main()
    