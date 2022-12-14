library ieee;

use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity prog_counter is
	port(
		i_clkin : in std_logic;
		i_reset : in std_logic;
        i_data_bus_in : in std_logic_vector(7 downto 0); --All th 8 bits are used but only the 4 LSB are considered
		o_data_bus_out : out std_logic_vector(7 downto 0)
    );
end entity prog_counter;

architecture rtl of prog_counter is

	constant c_PC_MAX : natural := 256;
    signal r_PC_REG : natural range 0 to c_PC_MAX;

    --Process to increment the counter and jump the counter value
    p_counter : process(i_clkin,i_reset)
        begin
            if i_reset = '0' then
                r_PC_REG <= 0; --Resetting the count register for program counter
            elsif rising_edge(i_clkin) then
                if r_PC_REG = c_PC_MAX-1 then
                    r_PC_REG <= 0;
                else
                    r_PC_REG <= r_PC_REG+1;
                end if;
            end if;
    end process p_counter;

    o_data_bus_out <= std_logic_vector(to_unsigned(r_PC_REG,o_data_bus_out'length))

end rtl;